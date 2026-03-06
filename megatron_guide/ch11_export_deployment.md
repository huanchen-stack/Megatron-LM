# Chapter 11 — Model Export & Deployment

> **Prerequisites**: Chapter 7 (Checkpointing), Chapter 10 (Quantization/Pruning). This chapter covers the last mile: getting your trained model from Megatron into a production inference server.

---

## 11.1 The Deployment Gap

You've trained, aligned, and quantized your model. Now you need to actually serve it. But there's a problem:

- **Megatron checkpoints** are optimized for *training* (sharded across TP/PP/EP ranks, with optimizer states)
- **Inference runtimes** need checkpoints optimized for *serving* (consolidated, quantized, engine-compiled)

The **export** step bridges this gap:

```
Megatron Distributed Checkpoint
  (TP=8, PP=4, EP=64)
         │
         ▼
    ┌─────────┐
    │  Export  │  ← ModelOpt: export_mcore_gpt_to_hf()
    └────┬────┘
         │
         ▼
  HuggingFace-Format
  Unified Checkpoint
    (single directory)
         │
    ┌────┼────┬─────────┐
    ▼    ▼    ▼         ▼
 TRT-LLM  vLLM  SGLang  Other
```

---

## 11.2 Export Targets — Where Models Go

### TensorRT-LLM (TRT-LLM)

NVIDIA's high-performance inference library:
- Compiles models into optimized TensorRT engines
- FP8/FP4 kernels on H100/Blackwell
- Supports speculative decoding (EAGLE3, Medusa)
- KV cache quantization
- Highest throughput on NVIDIA hardware

### vLLM

Open-source inference engine:
- PagedAttention for efficient KV cache management
- Continuous batching
- Good community support
- Supports quantized models via ModelOpt integration

### SGLang

Emerging inference framework:
- Optimized for structured generation (JSON mode, regex constraints)
- RadixAttention for KV cache sharing
- Fast for multi-turn conversations

### Direct Megatron Inference

Megatron also has a built-in inference engine (`megatron/core/inference/`):
- Used for RL training (Chapter 9) where inference and training share the same process
- Supports CUDA graphs for low-latency generation
- Not typically used for standalone deployment

---

## 11.3 The ModelOpt Export Pipeline

### Export Script (`export.py`)

```python
# From examples/post_training/modelopt/export.py

# 1. Initialize Megatron and load the (quantized) checkpoint
initialize_megatron(...)
model = get_model(model_provider, wrap_with_ddp=False)

# 2. Materialize model from meta device (for low-memory loading)
unwrapped_model = unwrap_model(model)[0]
unwrapped_model.to_empty(device="cpu")

# 3. Load the ModelOpt checkpoint (includes quantization state)
load_modelopt_checkpoint(model)

# 4. Export to HuggingFace-format unified checkpoint
mtex.export_mcore_gpt_to_hf(
    unwrapped_model,
    pretrained_model_name,        # e.g., "meta-llama/Llama-3.2-1B-Instruct"
    export_extra_modules=True,     # Include EAGLE3/Medusa heads
    dtype=torch.bfloat16,
    export_dir="/path/to/output",
    moe_router_dtype=model.config.moe_router_dtype,
)
```

### What Happens During Export

1. **Weight Collection**: Gathers weights from all TP/PP/EP ranks into a single consolidated set
2. **Format Translation**: Maps Megatron layer names to HuggingFace conventions
3. **Quantization Metadata**: Preserves scale factors and quantization configs
4. **Extra Modules**: Exports EAGLE3 draft heads or Medusa heads if present
5. **Config Generation**: Creates `config.json` compatible with the target runtime

### The Meta-Device Trick

For very large models, export uses **meta-device initialization** to avoid OOM:

```python
# Instead of allocating the full model in GPU memory:
args.use_cpu_initialization = True
args.init_model_with_meta_device = True

# The model is created with no memory allocation
model = get_model(model_provider, wrap_with_ddp=False)

# Then materialized on CPU, loaded shard by shard
unwrapped_model.to_empty(device="cpu")
load_modelopt_checkpoint(model)  # Fills in actual weights
```

This lets you export a 405B-parameter model without needing 405B parameters worth of GPU memory.

---

## 11.4 Megatron's Built-in TRT-LLM Export

Megatron Core also has a native TRT-LLM export path in `megatron/core/export/trtllm/`:

### TRTLLMHelper

```python
# megatron/core/export/trtllm/trtllm_helper.py

class TRTLLMHelper:
    """Convert Megatron model to TRT-LLM format and optionally build engines."""
    
    def __init__(self, transformer_config, model_type, ...):
        # Maps Megatron layer names → TRT-LLM layer names
        # e.g., "decoder.layers.0.self_attention.linear_qkv.weight"
        #     → TRT-LLM's expected naming convention
    
    def get_trtllm_pretrained_config_and_model_weights(
        self, model_state_dict, dtype, ...
    ):
        """Convert model weights to TRT-LLM format."""
        # 1. Create TRT-LLM PretrainedConfig
        # 2. Convert and remap all weight tensors
        # 3. Handle TP/PP resharding for inference parallelism
        return config, weights
    
    def build_and_save_engine(self, ...):
        """Compile TRT-LLM engine from converted weights."""
        # Uses TRT-LLM's build API to create optimized engines
```

### Weight Conversion

The conversion handles non-trivial mappings:

```
Megatron (training format)           TRT-LLM (inference format)
─────────────────────────           ─────────────────────────
linear_qkv.weight                 → attn.qkv.weight (merged QKV)
linear_proj.weight                → attn.dense.weight
linear_fc1.weight                 → mlp.fc.weight (or gate+up merged)
linear_fc2.weight                 → mlp.proj.weight
input_layernorm.weight            → input_layernorm.weight
pre_mlp_layernorm.weight          → post_layernorm.weight
```

For quantized models, additional scale factors are exported:
- Per-channel weight scales
- Per-tensor activation scales  
- KV cache quantization parameters

---

## 11.5 The Megatron Inference Engine

While ModelOpt export targets external runtimes, Megatron has its own inference engine in `megatron/core/inference/`:

### Architecture

```
┌──────────────────────────────────────┐
│        Text Generation Server        │
│    (HTTP/gRPC, OpenAI-compatible)    │
├──────────────────────────────────────┤
│     Text Generation Controller       │
│  (sampling, beam search, streaming)  │
├──────────────────────────────────────┤
│          Inference Engine             │
│  (batching, scheduling, execution)   │
├──────────────────────────────────────┤
│       Model Inference Wrapper        │
│  (adapts GPTModel for inference)     │
├──────────────────────────────────────┤
│           Scheduler                  │
│  (manages request queue, batching)   │
└──────────────────────────────────────┘
```

### Key Components

**Scheduler** (`scheduler.py`):
- Manages incoming requests
- Continuous batching (add/remove sequences dynamically)
- KV cache management

**Inference Engine** (`engines/`):
- Runs the model forward pass
- Supports CUDA graphs for low-latency
- Data-parallel inference coordination

**Sampling** (`sampling_params.py`, `common_inference_params.py`):
- Temperature, top-k, top-p sampling
- Repetition penalty
- Stop sequences

**Unified Memory** (`unified_memory.py`):
- UVM support for offloading KV cache or model weights
- Critical for RL training where inference and training share GPU memory

This inference engine is primarily used for:
1. **RL training** (Chapter 9) — generating rollouts during GRPO
2. **Evaluation** — running benchmarks during training
3. **Development/debugging** — quick inference without external servers

---

## 11.6 End-to-End Deployment Examples

### Example 1: Quantize and Deploy Llama 3.2

```bash
# Step 1: Quantize to NVFP4
TP=1 \
HF_MODEL_CKPT=/models/Llama-3.2-1B-Instruct \
MLM_MODEL_SAVE=/tmp/llama-quant \
./quantize.sh meta-llama/Llama-3.2-1B-Instruct NVFP4_DEFAULT_CFG

# Step 2: Export to unified checkpoint
PP=1 \
HF_MODEL_CKPT=/models/Llama-3.2-1B-Instruct \
MLM_MODEL_CKPT=/tmp/llama-quant \
EXPORT_DIR=/tmp/llama-export \
./export.sh meta-llama/Llama-3.2-1B-Instruct

# Step 3: Deploy with TRT-LLM
trtllm-serve /tmp/llama-export \
    --host 0.0.0.0 --port 8000 \
    --backend pytorch \
    --tp_size 1
```

### Example 2: Prune → Distill → Quantize → Deploy

```bash
# Step 1: Prune Qwen3-8B from 36 to 24 layers
PP=1 TARGET_NUM_LAYERS=24 \
HF_MODEL_CKPT=/models/Qwen3-8B \
MLM_MODEL_SAVE=/tmp/qwen-pruned \
./prune.sh Qwen/Qwen3-8B

# Step 2: Recover via distillation from original 36-layer model
MLM_EXTRA_ARGS="--num-layers 24 --export-kd-teacher-load /models/Qwen3-8B" \
./finetune.sh Qwen/Qwen3-8B

# Step 3: Quantize the distilled model
MLM_MODEL_CKPT=/tmp/qwen-pruned \
./quantize.sh Qwen/Qwen3-8B FP8_DEFAULT_CFG

# Step 4: Export and deploy
./export.sh Qwen/Qwen3-8B
```

### Example 3: EAGLE3 Speculative Decoding

```bash
# Step 1: Train EAGLE3 draft model
TP=1 \
HF_MODEL_CKPT=/models/Llama-3.2-1B-Instruct \
MLM_MODEL_SAVE=/tmp/llama-eagle3 \
./eagle3.sh meta-llama/Llama-3.2-1B-Instruct

# Step 2: Export EAGLE3 checkpoint
PP=1 \
HF_MODEL_CKPT=/models/Llama-3.2-1B-Instruct \
MLM_MODEL_CKPT=/tmp/llama-eagle3 \
EXPORT_DIR=/tmp/llama-eagle3-export \
./export.sh meta-llama/Llama-3.2-1B-Instruct

# Step 3: Deploy with speculative decoding
trtllm-serve /tmp/llama-eagle3-export \
    --host 0.0.0.0 --port 8000 \
    --backend pytorch \
    --tp_size 1 \
    --extra_llm_api_options eagle3_config.yml
```

Where `eagle3_config.yml`:
```yaml
speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: /tmp/llama-eagle3-export
```

---

## 11.7 Checkpoint Format Landscape

Understanding the checkpoint ecosystem:

```
                    ┌─────────────────────┐
                    │   Training Formats   │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                   ▼
    ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
    │   Megatron    │  │  HuggingFace  │  │   DeepSpeed  │
    │ Distributed   │  │    (safetens) │  │   (ZeRO)     │
    │  Checkpoint   │  │               │  │              │
    └───────┬───────┘  └───────┬───────┘  └──────┬───────┘
            │                  │                  │
            ▼                  ▼                  ▼
    ┌────────────────────────────────────────────────────┐
    │              Megatron Bridge / ModelOpt              │
    │         (Bidirectional Checkpoint Conversion)        │
    └───────────────────────┬────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌────────────┐ ┌──────────────┐
    │  TRT-LLM     │ │   vLLM     │ │   SGLang     │
    │  Engine      │ │  (GPTQ/    │ │  (AWQ/       │
    │              │ │   AWQ/FP8) │ │   FP8)       │
    └──────────────┘ └────────────┘ └──────────────┘
```

### Format Details

| Format | Used By | Key Properties |
|--------|---------|---------------|
| Megatron Distributed | Megatron training | ShardedTensor, supports resharding, includes optimizer |
| HuggingFace safetensors | HF ecosystem | Single-shard or model-parallel, widely compatible |
| TRT-LLM Engine | TRT-LLM serving | Compiled engine, hardware-specific, fastest inference |
| Unified Checkpoint | ModelOpt export | HF-like format enriched with quantization metadata |

### Megatron Bridge

[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) provides bidirectional conversion between HuggingFace and Megatron formats:

```
HuggingFace → Megatron: for training an open-source model
Megatron → HuggingFace: for deploying a trained model
```

This is what ModelOpt uses internally via `import_mcore_gpt_from_hf()` and `export_mcore_gpt_to_hf()`.

---

## 11.8 Key Files Covered

| File | Purpose |
|------|---------|
| `examples/post_training/modelopt/export.py` | Main export script: Megatron → HF-format |
| `examples/post_training/modelopt/export.sh` | Shell wrapper for export pipeline |
| `examples/post_training/modelopt/ADVANCED.md` | Multi-node/Slurm deployment examples |
| `megatron/core/export/export_config.py` | `ExportConfig` dataclass (TP/PP for inference) |
| `megatron/core/export/trtllm/trtllm_helper.py` | `TRTLLMHelper`: native Megatron→TRT-LLM conversion |
| `megatron/core/export/trtllm/trtllm_layers.py` | Layer name mapping for TRT-LLM |
| `megatron/core/export/trtllm/trtllm_weights_converter/` | Weight format conversion utilities |
| `megatron/core/inference/` | Megatron's built-in inference engine |
| `megatron/core/inference/engines/` | Inference execution engines |
| `megatron/core/inference/scheduler.py` | Request scheduling and batching |
| `megatron/core/inference/text_generation_server/` | HTTP server for text generation |

---

## 11.9 Interview-Ready Takeaways

### The Export Pipeline

> "After training in Megatron (which uses distributed checkpoints sharded across TP/PP/EP), you export to a unified HuggingFace-format checkpoint that inference runtimes can consume. ModelOpt's export handles weight consolidation, name remapping, and quantization metadata preservation."

### Why Not Serve Directly from Megatron?

> "Megatron's model format is optimized for training — weights are sharded across GPUs in ways that minimize communication during gradient computation. Inference runtimes shard differently (optimizing for throughput and latency), so you need to reshard. Also, inference engines like TRT-LLM compile the model into CUDA kernels that are much faster than PyTorch eager mode."

### TRT-LLM vs vLLM vs SGLang

| Feature | TRT-LLM | vLLM | SGLang |
|---------|---------|------|--------|
| Performance | Highest on NVIDIA | Very good | Good |
| Quantization | FP8, FP4, INT4 | GPTQ, AWQ, FP8 | AWQ, FP8 |
| Speculative Decoding | EAGLE3, Medusa | Medusa, Eagle | Eagle |
| Structured Output | Basic | Basic | Best (RadixAttention) |
| Setup Complexity | Higher | Lower | Lower |

### Deployment Checklist

> "For production deployment: (1) Choose your target precision (FP8 for quality, FP4 for throughput), (2) Quantize with calibration data representative of production traffic, (3) Run accuracy evaluation (MMLU, MT-Bench) before and after quantization, (4) Export to your target runtime, (5) Load test to verify throughput and latency SLOs, (6) Enable speculative decoding if latency-sensitive."

### Meta-Device Initialization

> "For very large models (100B+), export can OOM if you naively load the full model. Meta-device initialization creates the model skeleton without allocating memory, then fills in weights shard by shard from the checkpoint. This lets you export models much larger than your GPU memory."

### The Full Lifecycle — One Sentence

> "Data preprocessing → pretraining → SFT → RLHF/GRPO → optional pruning/distillation → quantization → export → deploy: Megatron handles the first five steps, ModelOpt handles compression and export, and TRT-LLM/vLLM/SGLang handle serving."
