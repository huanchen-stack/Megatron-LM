# Chapter 10 — Post-Training: Quantization, Pruning & Distillation

> **Prerequisites**: Chapters 3 (Model Architecture), 6 (Mixed Precision), 7 (Checkpointing). This chapter covers how to compress a trained model for efficient deployment.

---

## 10.1 Why Compress a Model?

You've spent weeks pretraining and post-training a 70B-parameter model. It works great — but serving it costs $50/hour on 8× H100 GPUs. Can we make it smaller and faster without losing too much quality?

That's the job of **model compression**:

| Technique | What It Does | Typical Savings |
|-----------|-------------|-----------------|
| **Quantization** | Reduce weight/activation precision (BF16→FP8→FP4) | 2-4× memory, 1.5-3× throughput |
| **Pruning** | Remove unnecessary neurons/layers/experts | 20-60% fewer parameters |
| **Knowledge Distillation** | Transfer knowledge from large→small model | Train better small models |

These techniques are often **combined**: prune → distill → quantize → deploy.

### The ModelOpt Pipeline

Megatron integrates with NVIDIA's **Model Optimizer** (`nvidia-modelopt`) for all three:

```
HuggingFace Checkpoint
        │
        ▼
┌─────────────────┐
│  Megatron Model  │ ← on-the-fly HF→Megatron conversion
└───────┬─────────┘
        │
   ┌────┼────┬────────────┐
   ▼    ▼    ▼            ▼
Quantize  Prune  Distill  EAGLE3 (speculative)
   │    │    │            │
   ▼    ▼    ▼            ▼
  Fine-tune (optional QAT/recovery)
        │
        ▼
┌─────────────────┐
│   Export to HF   │ → TensorRT-LLM / vLLM / SGLang
└─────────────────┘
```

---

## 10.2 Quantization — The Complete Picture

### What Is Quantization?

Quantization maps high-precision floating-point numbers to lower-precision representations:

```
BF16 (16 bits) → FP8 (8 bits)  → FP4 (4 bits) → INT4 (4 bits)
   1.0×             0.5×            0.25×            0.25×
   memory            memory          memory           memory
```

### Types of Quantization

**1. Post-Training Quantization (PTQ)**
- Apply quantization to a trained model without retraining
- Fast (minutes to hours) but may lose some accuracy
- Uses calibration data to determine optimal scale factors

**2. Quantization-Aware Training (QAT)**
- Simulate quantization during training
- "Fake quantization": forward pass uses low precision, backward pass uses full precision
- Better accuracy than PTQ but requires training compute

**3. Weight-Only vs Weight-and-Activation**
- **Weight-only**: only quantize stored weights (simpler, always safe)
- **Weight + Activation**: also quantize activations during computation (faster matmuls on supported hardware)

### Quantization Formats in Megatron/ModelOpt

| Format | Weight Bits | Activation Bits | Hardware | Notes |
|--------|------------|-----------------|----------|-------|
| `FP8_DEFAULT_CFG` | E4M3 (8-bit) | E4M3 (8-bit) | H100+ | Good accuracy-speed tradeoff |
| `NVFP4_DEFAULT_CFG` | E2M1 (4-bit) | E4M3 (8-bit) | Blackwell+ | Best for Blackwell GPUs |
| `INT8_SQ_DEFAULT_CFG` | INT8 | INT8 | A100+ | SmoothQuant |
| `W4A8_AWQ_BETA_CFG` | INT4 (AWQ) | FP8 | H100+ | Aggressive compression |

### How PTQ Works in Megatron

```python
# From quantize.py — the core PTQ flow:

# 1. Build the Megatron model
model = get_model(model_provider, wrap_with_ddp=False)

# 2. Import HuggingFace weights into Megatron format
import_mcore_gpt_from_hf(unwrapped_model, hf_checkpoint_path, workspace_dir)

# 3. Get quantization config
mtq_config = QUANT_CFG_CHOICES["NVFP4_DEFAULT_CFG"]

# 4. Calibrate and quantize (runs calibration data through the model)
mtq.quantize(unwrapped_model, mtq_config, calibration_forward_loop)

# 5. Optionally compress to real low-bit (not fake-quant)
if args.compress:
    mtq.compress(unwrapped_model)  # Actually pack weights into low-bit

# 6. Save Megatron checkpoint (resumable for QAT or export)
save_checkpoint(1, model, None, None, 0, release=True)
```

### Calibration — Why It Matters

Quantization needs to know the **range** of values each tensor will see. Calibration runs representative data through the model to collect these statistics:

```python
def calibration_forward_loop(model):
    dataloader = get_calib_dataloader(
        dataset_path_or_name="cnn_dailymail",
        tokenizer=tokenizer,
        calib_size=512,          # 512 calibration samples
        max_sequence_length=512,
    )
    for sample in dataloader:
        simple_generate(model, sample["input_ids"], osl=1, calibration_mode=True)
```

During calibration, ModelOpt records:
- **Per-channel min/max** for weights
- **Per-tensor or per-token** statistics for activations
- Optimal **scale factors** and **zero points**

### KV Cache Quantization

In addition to weight/activation quantization, the **KV cache** can also be quantized to save memory during inference:

```bash
# Add KV cache quantization to the quantize command
MLM_EXTRA_ARGS="--export-kv-cache-quant fp8" ./quantize.sh ...
```

Options: `fp8`, `fp8_affine`, `nvfp4`, `nvfp4_affine`, `nvfp4_rotate`

This is particularly impactful for long-context inference where the KV cache dominates memory.

### Selective Quantization

You can skip quantization for sensitive layers:

```python
# Skip first N layers (often more sensitive to quantization)
--num-first-layers-to-skip-quant 2

# Skip last N layers  
--num-last-layers-to-skip-quant 1

# Disable QKV quantization (attention is often sensitive)
--disable-qkv-quant
```

---

## 10.3 Pruning — Making Models Smaller

### What Is Pruning?

Pruning removes parts of the model that contribute least to its performance. Unlike quantization (which keeps the same architecture but reduces precision), pruning **changes the architecture**.

### Pruning Dimensions

Megatron/ModelOpt supports pruning along many dimensions:

| Dimension | What It Removes | Example |
|-----------|----------------|---------|
| `TARGET_NUM_LAYERS` | Entire transformer layers | 36 → 24 layers |
| `TARGET_FFN_HIDDEN_SIZE` | MLP neurons | 14336 → 10240 |
| `TARGET_HIDDEN_SIZE` | Embedding dimension | 4096 → 3072 |
| `TARGET_NUM_ATTENTION_HEADS` | Attention heads | 32 → 24 |
| `TARGET_NUM_QUERY_GROUPS` | GQA groups | 8 → 6 |
| `TARGET_NUM_MOE_EXPERTS` | MoE experts | 64 → 48 |
| `LAYERS_TO_DROP` | Specific layers by index | Drop layers 12, 13, 14 |

### How Pruning Works

**Step 1: Importance Scoring**

Run calibration data through the model and compute importance scores:

```python
# From prune.py:
mtp.prune(
    unwrapped_model,
    mode="mcore_minitron",          # NVIDIA's Minitron pruning method
    constraints={"export_config": {  # Target architecture
        "num_layers": 24,
        "ffn_hidden_size": 10240,
    }},
    config={"forward_loop": calibration_loop},
)
```

For **depth pruning** (removing layers), it uses the **Block Influence** metric from [Shortened LLaMA](https://arxiv.org/abs/2403.03853):
- Compute cosine similarity between each layer's input and output
- Layers with highest similarity (input ≈ output) contribute least → prune them

For **width pruning** (reducing dimensions), it uses activation-based importance scores:
- Track which neurons/heads are activated most frequently
- Remove the least important ones

**Step 2: Architecture Modification**

After scoring, the model architecture is physically modified:
- Layers are removed
- Linear layers are resized
- Embeddings are sliced

**Step 3: Recovery Training**

Pruning hurts accuracy. Recovery is done via fine-tuning or distillation:

```bash
# Prune
./prune.sh Qwen/Qwen3-8B  # 36 → 24 layers

# Recover via fine-tuning
MLM_EXTRA_ARGS="--num-layers 24" ./finetune.sh Qwen/Qwen3-8B
```

### The Minitron Approach (NVIDIA)

NVIDIA's [Minitron](https://arxiv.org/abs/2407.14679) is the pruning methodology used in ModelOpt:

1. **Prune width and depth** using importance scores
2. **Distill** from the original model to recover quality
3. Achieves models like **Nemotron-Nano** (9B from a larger teacher)

---

## 10.4 Knowledge Distillation — Learning from a Teacher

### What Is Knowledge Distillation?

A large **teacher** model transfers its knowledge to a smaller **student** model by training the student to match the teacher's behavior:

```
Teacher (70B, frozen) ──── soft logits ────▶ KL-Divergence Loss
                                              │
Student (8B, trainable) ── soft logits ───────┘
```

### Why Not Just Train the Student from Scratch?

The teacher's **soft logits** contain richer information than hard labels:
- For the input "The capital of France is ___", hard label says only "Paris"
- Soft logits say: "Paris" (0.85), "Lyon" (0.03), "Marseille" (0.02), "Berlin" (0.01), ...
- This "dark knowledge" teaches the student about relationships between tokens

### Distillation in Megatron

Megatron uses ModelOpt's `DistillationModel` wrapper:

```python
# The model builder wraps student + teacher:
# megatron/post_training/model_builder.py creates:
#   DistillationModel(student=student_gpt, teacher=teacher_gpt)
```

Configuration is via YAML:

```yaml
# distillation config
logit_layers: ["output_layer", "output_layer"]  # student, teacher layer names
intermediate_layer_pairs:
  - ["decoder.layers.0.input_layernorm", "decoder.layers.0.input_layernorm"]
  - ["decoder.final_layernorm", "decoder.layers.30.input_layernorm"]
skip_lm_loss: true            # Only use KD loss, skip standard LM loss
kd_loss_scale: 10.0           # Scale factor for distillation loss
logit_kl_temperature: 1.0     # Temperature for softening distributions
```

### Loss Functions

**Logit Distillation** (always used):
```
L_logit = KL(softmax(z_teacher/T), softmax(z_student/T)) × T²
```
Where T is the temperature — higher T makes distributions softer.

**Intermediate Layer Distillation** (optional):
```
L_intermediate = 1 - cos_sim(h_teacher, h_student)
```
Aligns hidden representations at specific layers between teacher and student.

**Combined Loss**:
```
L_total = L_LM + kd_loss_scale × (L_logit + L_intermediate)
```

Or if `skip_lm_loss: true`:
```
L_total = kd_loss_scale × (L_logit + L_intermediate)
```

### Distillation Workflow

```bash
# 1. Start with student and teacher checkpoints
# Teacher: full 70B model
# Student: pruned 8B model (or trained from scratch)

# 2. Run distillation
python pretrain_gpt.py \
    --export-kd-teacher-load /path/to/teacher_checkpoint \
    --export-te-mcore-model \
    --export-kd-distill-cfg /path/to/distill_config.yaml
```

The teacher model is **frozen** — only the student receives gradient updates.

---

## 10.5 Putting It All Together — The Compression Pipeline

### Typical Production Pipeline

```
1. Start with pretrained 70B model (HuggingFace)
         │
2. Prune to 8B  (depth: 80→32 layers, width: smaller FFN)
         │
3. Distill from 70B teacher → 8B student (recover quality)
         │
4. SFT the distilled 8B model (Chapter 8)
         │
5. RLHF/GRPO the SFT'd model (Chapter 9)
         │
6. Quantize to NVFP4 (final compression for deployment)
         │
7. Export to TensorRT-LLM unified checkpoint
         │
8. Deploy (Chapter 11)
```

### Quantization-Aware Training (QAT) Sub-Pipeline

For maximum quality at low precision:

```bash
# Step 1: Initial quantization (fake-quant)
./quantize.sh meta-llama/Llama-3.2-1B-Instruct NVFP4_DEFAULT_CFG

# Step 2: QAT fine-tuning (trains with fake-quant enabled)
./finetune.sh meta-llama/Llama-3.2-1B-Instruct

# Step 3: Export compressed model
./export.sh meta-llama/Llama-3.2-1B-Instruct
```

During QAT fine-tuning, `finetune.py` uses the same `pretrain()` loop as pretraining (Chapter 4) but with:
- Quantized model loaded (quantization nodes simulate low-precision)
- SFT dataset (same packing mechanism as Chapter 8)
- Standard cross-entropy loss with answer-only masking

---

## 10.6 EAGLE3 — Speculative Decoding (Bonus)

While not compression per se, **speculative decoding** is a deployment optimization closely integrated with the ModelOpt pipeline.

### The Problem

LLM inference is **memory-bandwidth bound**, not compute-bound. Each token generation requires loading the entire model from memory but only does a tiny amount of computation. The GPU is mostly idle.

### The Solution: Draft-Then-Verify

1. A small **draft model** quickly generates K candidate tokens
2. The large **target model** verifies all K tokens in a single forward pass
3. Accept the longest prefix of correct tokens

```
Draft model:  "The capital of France is Paris , a beautiful city"
Target model: "The capital of France is Paris , a beautiful"  ✓ (accepted 7 tokens)
                                                    "town"   ✗ (rejected, resample)
```

This is **lossless** — the output distribution is identical to the target model, but you generate multiple tokens per forward pass.

### EAGLE3 in Megatron

Megatron supports training EAGLE3 draft models through ModelOpt:

**Online Training** (recommended):
```bash
# Both target (frozen) and draft models in memory
# Hidden states generated on-the-fly
./eagle3.sh meta-llama/Llama-3.2-1B-Instruct
```

**Offline Training**:
```bash
# Step 1: Extract target model hidden states to disk
./offline_feature_extract.sh ...

# Step 2: Train draft model using saved features
./finetune.sh ... --export-offline-model --offline-distillation-data /path/to/features
```

The draft model learns to predict the target model's hidden states, which it then uses to predict the next tokens. The key metric is **Acceptance Length (AL)** — how many tokens are accepted on average per draft.

---

## 10.7 Key Files Covered

| File | Purpose |
|------|---------|
| `examples/post_training/modelopt/quantize.py` | PTQ pipeline: calibrate → quantize → compress → save |
| `examples/post_training/modelopt/prune.py` | Pruning pipeline: score → prune → save |
| `examples/post_training/modelopt/finetune.py` | QAT/recovery fine-tuning with SFT data |
| `examples/post_training/modelopt/export.py` | Export to HF-format for TRT-LLM/vLLM/SGLang |
| `examples/post_training/modelopt/convert_model.py` | Model conversion (add EAGLE/Medusa heads) |
| `examples/post_training/modelopt/distillation.md` | Knowledge distillation documentation |
| `examples/post_training/modelopt/speculative.md` | EAGLE3/Medusa speculative decoding docs |
| `examples/post_training/modelopt/README.md` | Support matrix and getting started guide |
| `megatron/post_training/model_builder.py` | ModelOpt model builder (wraps DistillationModel) |
| `megatron/post_training/loss_func.py` | Loss function for post-training (SFT + KD) |
| `megatron/post_training/arguments.py` | ModelOpt-specific argument definitions |
| `megatron/post_training/checkpointing.py` | ModelOpt checkpoint loading |

---

## 10.8 Interview-Ready Takeaways

### Quantization Fundamentals

> "Quantization reduces numerical precision of model weights and/or activations. PTQ is fast but may lose accuracy; QAT simulates quantization during training for better results. The key design choice is which format (FP8, FP4, INT4) and whether to quantize activations too."

### Calibration

> "Calibration runs representative data through the model to determine the range of values each tensor will see. This lets us set optimal scale factors. Bad calibration data → bad quantization quality."

### When to Use What

| Goal | Technique | Quality Impact |
|------|-----------|---------------|
| 2× faster, minimal quality loss | FP8 PTQ | <1% degradation |
| 4× smaller for edge deployment | FP4/INT4 PTQ + QAT | 1-3% degradation |
| Create a smaller model variant | Pruning + Distillation | 2-5% degradation |
| Maximum throughput | Quantize + Speculative decoding | Near lossless |

### Pruning vs Distillation

> "Pruning removes parts of a model (structural change). Distillation trains a small model to mimic a large one (knowledge transfer). In practice, you prune first, then distill from the original to recover quality. This is the Minitron approach."

### The Temperature Trick in Distillation

> "Higher temperature makes the teacher's soft logits more informative. At T=1, the distribution is peaked on the top token. At T=5, it spreads probability across many tokens, revealing the teacher's 'uncertainty' and knowledge about token relationships."

### EAGLE3 / Speculative Decoding

> "LLM inference is memory-bandwidth bound, not compute-bound. Speculative decoding uses a cheap draft model to guess multiple tokens, then the real model verifies them in parallel. It's mathematically lossless — same output distribution, just faster. The key metric is Acceptance Length: how many draft tokens get accepted on average."

### The Full Pipeline Interview Answer

> "The production pipeline for a state-of-the-art model typically goes: pretrain → SFT → RLHF/GRPO → prune (optional) → distill (if pruned) → quantize → export → deploy with speculative decoding. Megatron handles pretraining through GRPO, then ModelOpt handles compression and export to inference runtimes like TensorRT-LLM."
