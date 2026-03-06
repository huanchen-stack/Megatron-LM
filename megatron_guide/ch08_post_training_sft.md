# Chapter 8 — Post-Training: Supervised Fine-Tuning (SFT) & LoRA

> **Goal**: Understand how Megatron-LM handles supervised fine-tuning — from data formatting through the SFT dataset, loss masking, conversation packing, and parameter-efficient fine-tuning (PEFT) with LoRA.

---

## 8.1 What Is Post-Training?

After pretraining a base language model on trillions of tokens, the model can predict the next token but can't follow instructions, hold conversations, or refuse harmful requests. **Post-training** transforms a base model into a useful assistant through several stages:

```
Base Model (pretrained)
    │
    ├── SFT (Supervised Fine-Tuning)     ← This chapter
    │   └── Train on human-written instruction/response pairs
    │
    ├── RLHF / GRPO (Reinforcement Learning)  ← Chapter 9
    │   └── Optimize for human preferences via reward signal
    │
    └── Optimization (Quantization, Pruning)  ← Chapter 10
        └── Compress model for deployment
```

SFT is the first and most critical post-training step. It teaches the model *how* to respond to user instructions.

---

## 8.2 SFT vs Pretraining: Key Differences

| Aspect | Pretraining | SFT |
|--------|------------|-----|
| **Data format** | Raw text documents | Structured conversations (system/user/assistant) |
| **Loss target** | Predict every token | Only predict assistant responses (mask prompts) |
| **Data volume** | Trillions of tokens | Thousands to millions of examples |
| **Learning rate** | Higher (e.g., 3e-4) | Much lower (e.g., 1e-5 to 5e-6) |
| **Epochs** | Usually < 1 | Often 1-3 passes over data |
| **Parameters** | All | All (full fine-tuning) or subset (LoRA/PEFT) |

---

## 8.3 The SFT Data Pipeline

### 8.3.1 Data Format

Megatron's SFT expects JSONL files where each line contains a `messages` list:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"},
    {"role": "assistant", "content": "The capital of Germany is Berlin."}
  ]
}
```

This is the standard **chat format** used by OpenAI, LLaMA, and most modern LLMs.

### 8.3.2 SFTLowLevelDataset

The low-level dataset simply loads the JSONL and returns the messages:

```python
class SFTLowLevelDataset:
    def __init__(self, dataset_path: str):
        from datasets import load_dataset
        self.dataset = load_dataset("json", data_files=dataset_path, split="all")
    
    def __getitem__(self, idx: int) -> list:
        return self.dataset[idx]["messages"]
```

### 8.3.3 SFTDataset — The Main Dataset Class

`SFTDataset` inherits from `MegatronDataset` and handles:
1. Tokenization of conversations
2. Loss masking (only train on assistant responses)
3. Conversation packing (fit multiple conversations into one sequence)
4. Position ID management for packed sequences

```python
class SFTDataset(MegatronDataset):
    def __getitem__(self, idx):
        # 1. Get raw conversation
        merged_conversations = self.dataset[self.indices[idx]]
        
        # 2. Split into individual conversations (each starts with "system")
        split_conversations = self._split_conversations(merged_conversations)
        
        # 3. Tokenize and pack conversations into one sequence
        for conversation in split_conversations:
            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True
            )
            pack_tokens.extend(tokens)
            pack_targets.extend(targets)
        
        # 4. Create loss mask
        loss_mask = torch.ones(pack_length)
        loss_mask[labels == pad] = 0.0          # mask padding
        loss_mask[labels == IGNORE_INDEX] = 0.0  # mask prompts
        
        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,      # cumulative sequence lengths for packing
            'max_seqlen': max_seqlen,
        }
```

---

## 8.4 Loss Masking — The Heart of SFT

### 8.4.1 Why Mask the Prompt?

During pretraining, the model learns from every token. During SFT, we only want to learn from the **assistant's responses**, not from the user's prompts or system messages. This is done via a **loss mask**.

```
System: You are a helpful assistant.
User: What is 2+2?
Assistant: The answer is 4.

Tokens:   [SYS] You are a helpful assistant [USR] What is 2+2? [ASST] The answer is 4 [EOS]
Loss mask: 0     0   0  0 0        0          0    0    0  0     1     1   1      1  1  1
                    ↑ prompts: masked                              ↑ response: loss computed
```

### 8.4.2 How Targets Work

The tokenizer's `tokenize_conversation()` method returns two aligned sequences:
- `tokens`: the input token IDs
- `targets`: the expected next token at each position

For prompt positions, `targets[i] = IGNORE_INDEX (-100)`, so the loss mask is set to 0.
For response positions, `targets[i]` is the actual next token, so the loss mask is 1.

### 8.4.3 The Loss Function

The SFT loss function in `megatron/post_training/loss_func.py`:

```python
def loss_func(loss_mask, output_tensor, model):
    loss_lm = _mask_loss(output_tensor, loss_mask)
    num_tokens = loss_mask.sum()  # count of non-masked tokens
    return loss, num_tokens, report

def _mask_loss(output_tensor, loss_mask):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)  # element-wise mask and sum
    return loss
```

The loss is then averaged across non-masked tokens (done by the training loop's `num_tokens` tracking).

---

## 8.5 Conversation Packing

### 8.5.1 The Problem

SFT conversations vary wildly in length — some are 50 tokens, others are 4000. If we pad every sequence to `seq_length`, we waste massive compute on padding tokens.

### 8.5.2 The Solution: Packing

Multiple conversations are concatenated into a single sequence until it reaches `seq_length`:

```
Sequence 1 (50 tokens) + Sequence 2 (200 tokens) + Sequence 3 (100 tokens) + [padding]
├─────────────────── packed into one seq_length sequence ──────────────────────┤
```

### 8.5.3 How Packing Works in Megatron

The `SFTDataset.__getitem__()` method packs greedily:

```python
for conversation in split_conversations:
    tokens, targets = tokenizer.tokenize_conversation(conversation, ...)
    pack_tokens.extend(tokens)
    pack_targets.extend(targets)
    cu_seqlens.append(len(pack_tokens))
    
    # Stop if we've filled the sequence
    if len(pack_tokens) >= pack_length + 1:
        # Truncate and break
        break

# Pad if we didn't fill the sequence
if len(pack_tokens) < pack_length + 1:
    pad_len = pack_length + 1 - len(pack_tokens)
    pack_tokens.extend([pad] * pad_len)
```

### 8.5.4 cu_seqlens: Tracking Boundaries

The `cu_seqlens` (cumulative sequence lengths) tensor tells the attention mechanism where each conversation starts and ends:

```
cu_seqlens = [0, 50, 250, 350, 512]
                │    │      │    │
                │    │      │    └── end of padding / sequence length
                │    │      └── end of conversation 3
                │    └── end of conversation 2
                └── start of conversation 1
```

This is used by FlashAttention to prevent cross-attention between different conversations in the same packed sequence.

### 8.5.5 Position IDs for Packing

Each conversation in a packed sequence gets its own position IDs starting from 0:

```
Conversation 1:  pos 0, 1, 2, ..., 49
Conversation 2:  pos 0, 1, 2, ..., 199
Conversation 3:  pos 0, 1, 2, ..., 99
Padding:         pos 100, 101, ..., 161  (continues from last conversation)
```

This ensures RoPE positional encodings are correct for each conversation.

---

## 8.6 Chat Templates and Tokenization

### 8.6.1 What Are Chat Templates?

Different models use different formats to structure conversations:

**ChatML (used by many models):**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

**LLaMA format:**
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello! [/INST] Hi there! </s>
```

### 8.6.2 Megatron's Tokenizer Interface

The tokenizer must implement `tokenize_conversation()` which:
1. Applies the chat template to format roles
2. Tokenizes the formatted text
3. Identifies which tokens are from the assistant (trainable) vs. the user/system (masked)
4. Returns `(tokens, targets)` where targets for non-assistant tokens are `IGNORE_INDEX`

This is typically done through the HuggingFace tokenizer's chat template system.

---

## 8.7 LoRA and Parameter-Efficient Fine-Tuning (PEFT)

### 8.7.1 The Problem Full Fine-Tuning Creates

Full fine-tuning updates all parameters. For a 70B model:
- Need 70B × 4 bytes = 280 GB for FP32 master weights
- Plus ~560 GB for Adam states
- Total: ~840 GB of optimizer state (even with ZeRO, this is huge)
- Risk of catastrophic forgetting

### 8.7.2 What Is LoRA?

**Low-Rank Adaptation (LoRA)** adds small trainable matrices alongside frozen pretrained weights:

```
Original:     y = W·x           (W is d_out × d_in, frozen)
With LoRA:    y = W·x + B·A·x   (A is r × d_in, B is d_out × r, trainable)
```

Where:
- `r` (rank) << min(d_in, d_out), typically 8-64
- `W` is frozen (no gradients, no optimizer state)
- Only `A` and `B` are trained

**Memory savings**: For a 70B model with LoRA rank 16:
- Trainable parameters: ~0.1% of total
- Optimizer state: ~0.1% of full fine-tuning
- GPU memory: can fine-tune 70B on a single 80GB GPU

### 8.7.3 LoRA Math

```
Forward:    h = (W + B·A) · x = W·x + (B·A)·x

Gradient:   ∂L/∂A = Bᵀ · ∂L/∂h · xᵀ
            ∂L/∂B = ∂L/∂h · (A·x)ᵀ

Merge:      W_merged = W + B·A   (for deployment, add LoRA to base weights)
```

**Initialization**: `A` is initialized with Kaiming/Gaussian, `B` is initialized to zero. This ensures the LoRA starts as an identity transformation (no change to pretrained behavior).

### 8.7.4 LoRA in the Megatron Ecosystem

Megatron-LM itself does not have a built-in LoRA module at the `megatron/core` level. Instead, LoRA is supported through:

1. **NVIDIA NeMo Framework**: NeMo wraps Megatron-Core and provides full LoRA/PEFT support with proper distributed training integration
2. **NVIDIA ModelOpt**: Can apply LoRA during quantization-aware training
3. **Manual implementation**: Users can freeze base parameters and add LoRA modules

**Note on `q_lora_rank` / `kv_lora_rank`**: These parameters in `TransformerConfig` refer to DeepSeek's **Multi-Latent Attention (MLA)**, which uses a similar low-rank compression technique for KV cache, not for fine-tuning. The naming is coincidental.

### 8.7.5 Other PEFT Methods

| Method | Trainable Params | How It Works |
|--------|-----------------|--------------|
| **LoRA** | ~0.1% | Low-rank matrices added to attention/MLP projections |
| **QLoRA** | ~0.1% | LoRA + 4-bit quantized base model |
| **Prefix Tuning** | ~0.1% | Learnable "prefix" tokens prepended to attention |
| **Adapter** | ~1-3% | Small bottleneck modules inserted between layers |
| **Full Fine-Tuning** | 100% | All parameters updated |

### 8.7.6 When to Use Each

| Scenario | Recommended Approach |
|----------|---------------------|
| Limited GPU memory | QLoRA |
| Need maximum quality | Full fine-tuning |
| Multiple tasks from one base | LoRA (one adapter per task) |
| Instruction following | LoRA with rank 16-64 |
| Domain adaptation | Full fine-tuning or LoRA with rank 64-256 |

---

## 8.8 Running SFT with Megatron

### 8.8.1 Data Preparation

```bash
# Prepare JSONL data in the correct format
# Each line: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
```

### 8.8.2 SFT Training Command

```bash
python pretrain_gpt.py \
    --finetune \                           # enable fine-tuning mode
    --load /path/to/pretrained/checkpoint \
    --save /path/to/sft/checkpoint \
    --data-path /path/to/sft_data.jsonl \
    --dataloader-type cyclic \
    --lr 1e-5 \                            # lower LR for fine-tuning
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-warmup-iters 100 \
    --train-iters 2000 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --bf16 \
    --use-dist-ckpt \
    --ckpt-format torch_dist
```

### 8.8.3 Key Arguments for SFT

| Argument | Typical Value | Purpose |
|----------|--------------|---------|
| `--finetune` | (flag) | Skip strict checkpoint validation, don't load optimizer state |
| `--lr` | 1e-5 to 5e-6 | Much lower than pretraining |
| `--lr-warmup-iters` | 50-200 | Short warmup |
| `--no-load-optim` | (flag) | Don't load optimizer from checkpoint |
| `--no-load-rng` | (flag) | Don't load RNG state from checkpoint |

---

## 8.9 Advanced SFT Topics

### 8.9.1 Multi-Turn Conversations

SFT datasets commonly include multi-turn conversations. The key challenge is loss masking across turns:

```
Turn 1: User → Assistant (train on assistant)
Turn 2: User → Assistant (train on assistant)
Turn 3: User → Assistant (train on assistant)
```

Each turn's user message is masked; each turn's assistant response has loss computed.

### 8.9.2 Knowledge Distillation During SFT

Megatron supports knowledge distillation (KD) during fine-tuning via ModelOpt:

```python
def loss_func(loss_mask, output_tensor, model):
    loss_lm = _mask_loss(output_tensor, loss_mask)
    
    if args.export_kd_teacher_load:
        losses = model.compute_kd_loss(
            student_loss=loss_lm,
            loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
        )
        loss = losses["kd_loss"]  # combined student + distillation loss
```

This trains a smaller "student" model to match a larger "teacher" model's outputs.

### 8.9.3 Context Parallelism with SFT

When using context parallelism (CP > 1), packed sequences need special padding to ensure each CP chunk has valid data:

```python
if self.config.context_parallel_size > 1:
    pad_granularity = self.config.context_parallel_size * 2
    mod_token_count = len(pack_tokens) % pad_granularity
    if mod_token_count != 0:
        pad_len = pad_granularity - mod_token_count
        extend_with_padding(pack_tokens, pack_targets, pack_positions, pad_len)
```

---

## 8.10 Key Files Covered

| File | Purpose |
|------|---------|
| `megatron/training/datasets/sft_dataset.py` | `SFTDataset`, `SFTLowLevelDataset` — SFT data pipeline |
| `megatron/post_training/loss_func.py` | `loss_func()`, `_mask_loss()` — loss masking |
| `megatron/post_training/model_builder.py` | `modelopt_gpt_mamba_builder()` — model creation with ModelOpt |
| `megatron/core/datasets/gpt_dataset.py` | `GPTDatasetConfig` — base config for dataset |
| `megatron/core/datasets/megatron_dataset.py` | `MegatronDataset` — base dataset class |

---

## 8.11 Interview-Ready Takeaways

1. **SFT teaches models to follow instructions** by training on conversation pairs. Unlike pretraining (predict every token), SFT only computes loss on **assistant responses** using a loss mask.

2. **Loss masking** sets prompt/system token losses to zero: `loss = sum(losses * loss_mask) / sum(loss_mask)`. The `IGNORE_INDEX = -100` sentinel in targets marks non-trainable tokens.

3. **Conversation packing** concatenates multiple short conversations into one `seq_length` sequence to avoid wasting compute on padding. `cu_seqlens` tracks boundaries for FlashAttention.

4. **Chat templates** (ChatML, LLaMA format) define how roles are formatted. The tokenizer's `tokenize_conversation()` applies the template and identifies assistant vs. non-assistant tokens.

5. **LoRA adds low-rank matrices** `B·A` alongside frozen weights `W`: `y = W·x + B·A·x`. With rank r, only `r × (d_in + d_out)` parameters are trainable instead of `d_in × d_out`. Typical r = 8-64.

6. **LoRA initialization**: `A` ~ Gaussian, `B` = 0. This ensures LoRA starts as an identity (no change to pretrained model). During deployment, LoRA can be **merged**: `W_new = W + B·A`.

7. **QLoRA** = LoRA on a 4-bit quantized base model. Enables fine-tuning 70B models on a single 80GB GPU.

8. **Megatron's `q_lora_rank`/`kv_lora_rank` ≠ LoRA fine-tuning**. These config parameters implement DeepSeek's Multi-Latent Attention (MLA), which uses low-rank compression of the KV cache to reduce inference memory.

9. **SFT hyperparameters** differ from pretraining: lower LR (1e-5 vs 3e-4), shorter training (1-3 epochs vs < 1 epoch), and often `--finetune --no-load-optim` flags.

10. **Knowledge distillation during SFT**: A teacher model's logits guide the student's training via KD loss = `α · CE_loss + (1-α) · KL(teacher || student)`, supported through ModelOpt integration.
