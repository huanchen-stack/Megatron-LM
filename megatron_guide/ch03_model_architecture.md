# Chapter 3: Model Architecture Deep Dive

## Overview

In Chapters 1–2 we turned raw text into token IDs and loaded them into GPU-friendly datasets. Now we need a **model** that consumes those tokens and produces next-token predictions. This chapter traces the entire model construction path in Megatron-LM — from the top-level `GPTModel` down to individual attention heads and MLP neurons — so you understand every layer of the architecture that will be trained.

The key insight of Megatron's architecture is the **Spec Pattern**: rather than hard-coding which PyTorch modules go inside each transformer layer, Megatron uses `ModuleSpec` objects (think of them as "blueprints") that describe *what* to build. This makes the model highly configurable — you can swap in Transformer Engine (TE) kernels, switch between standard attention and Multi-Latent Attention (MLA), or plug in Mixture-of-Experts (MoE) — all without changing the core model code.

---

## 3.1 The Big Picture: GPTModel → TransformerBlock → TransformerLayer

```
GPTModel
├── LanguageModelEmbedding        # Token + position embeddings
│   ├── word_embeddings           # VocabParallelEmbedding (TP-sharded)
│   └── position_embeddings       # (if learned_absolute)
├── RotaryEmbedding               # (if RoPE / YaRN / mRoPE)
├── TransformerBlock (decoder)    # The stack of transformer layers
│   ├── TransformerLayer[0]
│   │   ├── input_layernorm
│   │   ├── SelfAttention
│   │   │   ├── linear_qkv        # Fused QKV projection (ColumnParallelLinear)
│   │   │   ├── core_attention     # DotProductAttention or TE attention
│   │   │   └── linear_proj       # Output projection (RowParallelLinear)
│   │   ├── self_attn_bda         # Bias-Dropout-Add (fused)
│   │   ├── pre_mlp_layernorm
│   │   ├── MLP (or MoELayer)
│   │   │   ├── linear_fc1        # Up-projection (ColumnParallelLinear)
│   │   │   ├── activation_func   # SwiGLU / GeLU / etc.
│   │   │   └── linear_fc2        # Down-projection (RowParallelLinear)
│   │   └── mlp_bda               # Bias-Dropout-Add (fused)
│   ├── TransformerLayer[1]
│   │   └── ...
│   ├── ...
│   └── final_layernorm           # RMSNorm / LayerNorm at the end
├── output_layer                  # ColumnParallelLinear → vocab logits
└── (optional) MultiTokenPredictionBlock  # MTP heads for speculative decoding
```

**Entry point**: `pretrain_gpt.py` calls `model_provider()` which creates `GPTModel`.

**File**: `megatron/core/models/gpt/gpt_model.py`

---

## 3.2 GPTModel: The Top-Level Container

`GPTModel` extends `LanguageModule` and orchestrates three phases:

### Phase 1: Preprocessing (`_preprocess`)
1. **Embedding**: Converts `input_ids` → hidden states using `LanguageModelEmbedding`
2. **Positional Encoding**: Computes rotary embeddings (RoPE), YaRN, or multimodal RoPE

### Phase 2: Decoder (`self.decoder`)
3. Passes hidden states through `TransformerBlock` (the stack of layers)

### Phase 3: Postprocessing (`_postprocess`)
4. **Output Layer**: Projects hidden states → vocabulary logits via `ColumnParallelLinear`
5. **Loss**: Computes cross-entropy loss if labels are provided

```python
# Simplified GPTModel.forward()
def forward(self, input_ids, position_ids, attention_mask, labels=None, ...):
    # Phase 1: Embed tokens + compute rotary embeddings
    decoder_input, rotary_pos_emb, ... = self._preprocess(input_ids, position_ids, ...)
    
    # Phase 2: Pass through N transformer layers
    hidden_states = self.decoder(hidden_states=decoder_input, rotary_pos_emb=rotary_pos_emb, ...)
    
    # Phase 3: Project to vocab and compute loss
    logits, _ = self.output_layer(hidden_states)
    if labels is not None:
        loss = self.compute_language_model_loss(labels, logits)
        return loss
    return logits
```

### Pipeline Parallelism Flags

`GPTModel` accepts `pre_process` and `post_process` boolean flags:

| Flag | True | False |
|------|------|-------|
| `pre_process` | This rank has the embedding layer | Receives hidden states from previous PP stage |
| `post_process` | This rank has the output/loss layer | Sends hidden states to next PP stage |

In a 4-stage pipeline: Stage 0 has `pre_process=True`, Stage 3 has `post_process=True`, Stages 1–2 have both `False`.

---

## 3.3 The Spec Pattern: How Layers Are Configured

Before diving into individual components, let's understand the architecture's key design pattern.

### The Problem
Different deployments need different implementations:
- **Training with FP8**: Needs Transformer Engine (TE) kernels
- **Training without TE**: Needs pure Megatron-Core ("local") implementations  
- **Inference**: Needs optimized fused kernels
- **MoE models**: Need expert routing instead of dense MLP

### The Solution: ModuleSpec

A `ModuleSpec` is a dataclass that says "build module X with submodules Y":

```python
# From megatron/core/transformer/spec_utils.py
@dataclass
class ModuleSpec:
    module: type                    # The class to instantiate
    submodules: Any = None          # Nested specs for sub-components
    params: dict = field(...)       # Extra constructor kwargs
```

### Three Spec Providers

`gpt_layer_specs.py` provides three factory functions:

| Function | Backend | Use Case |
|----------|---------|----------|
| `get_gpt_layer_with_transformer_engine_spec()` | TE | FP8 training, fused kernels |
| `get_gpt_layer_local_spec()` | Megatron-Core only | CPU-friendly, no TE dependency |
| `get_gpt_layer_with_inference_spec()` | Inference-optimized | Serving with fused TP communication |

Each returns a `ModuleSpec` wrapping `TransformerLayer` with appropriate submodules:

```python
# Example: Local (no TE) spec
def get_gpt_layer_local_submodules(...) -> TransformerLayerSubmodules:
    return TransformerLayerSubmodules(
        input_layernorm=layer_norm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=backend.column_parallel_linear(),
                core_attention=backend.core_attention(),
                linear_proj=backend.row_parallel_linear(),
                q_layernorm=IdentityOp,   # No QK norm by default
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=layer_norm,
        mlp=ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
            ),
        ),
        mlp_bda=get_bias_dropout_add,
    )
```

### BackendSpecProvider

The actual linear/norm/attention module classes come from a `BackendSpecProvider`:

- `LocalSpecProvider` → Pure PyTorch / Megatron-Core modules
- `TESpecProvider` → Transformer Engine modules (with FP8 support)
- `InferenceSpecProvider` → Inference-optimized modules
- `KitchenSpecProvider` → Experimental "Kitchen" backend

This means the same `TransformerLayer` code works with any backend — only the *spec* changes.

---

## 3.4 TransformerBlock: The Layer Stack

**File**: `megatron/core/transformer/transformer_block.py`

`TransformerBlock` is simply a container that:
1. Builds N `TransformerLayer` instances from specs
2. Iterates through them in forward pass
3. Applies final layer norm
4. Handles activation checkpointing (recomputation)

### Layer Construction

```python
# Simplified from TransformerBlock._build_layers()
def _build_layers(self):
    self.layers = torch.nn.ModuleList([
        build_module(layer_spec, config=config, layer_number=i+1)
        for i, layer_spec in enumerate(self.submodules.layer_specs)
    ])
    
    if self.has_final_layernorm_in_this_stage():
        self.final_layernorm = LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
```

### Forward Pass (simplified)

```python
def forward(self, hidden_states, attention_mask, rotary_pos_emb=None, ...):
    for layer in self.layers:
        hidden_states, context = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            ...
        )
    
    if self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)
    
    return hidden_states
```

### Activation Checkpointing

For memory savings, `TransformerBlock` supports two recompute strategies:

| Method | Strategy | Memory | Compute Overhead |
|--------|----------|--------|-----------------|
| `uniform` | Checkpoint every N layers as a group | Medium savings | Medium overhead |
| `block` | Checkpoint first N individual layers | Fine-grained control | Varies |

Additionally, `selective` recomputation (configured via `recompute_modules`) allows checkpointing only specific operations within each layer (e.g., `core_attn`, `layernorm`, `mlp`).

---

## 3.5 TransformerLayer: The Fundamental Unit

**File**: `megatron/core/transformer/transformer_layer.py`

Each `TransformerLayer` contains 9 modules (some may be `IdentityOp` if unused):

```
Module 1: input_layernorm       → LayerNorm / RMSNorm
Module 2: self_attention        → SelfAttention (QKV projection + core attention + output projection)
Module 3: self_attn_bda         → Bias-Dropout-Add (fused residual connection)
Module 4: pre_cross_attn_layernorm → (IdentityOp for decoder-only models)
Module 5: cross_attention       → (IdentityOp for decoder-only models)
Module 6: cross_attn_bda        → (IdentityOp for decoder-only models)
Module 7: pre_mlp_layernorm     → LayerNorm / RMSNorm
Module 8: mlp                   → MLP or MoELayer
Module 9: mlp_bda               → Bias-Dropout-Add (fused residual connection)
```

### Forward Flow (Pre-Norm Architecture)

```
Input hidden_states
    │
    ├── residual = hidden_states
    │
    ▼
[input_layernorm] ──→ [SelfAttention] ──→ output + bias
    │                                          │
    │                   ┌──────────────────────┘
    ▼                   ▼
         [self_attn_bda]: hidden_states = dropout(output) + residual
    │
    ├── residual = hidden_states
    │
    ▼
[pre_mlp_layernorm] ──→ [MLP / MoE] ──→ output + bias
    │                                        │
    │                   ┌────────────────────┘
    ▼                   ▼
         [mlp_bda]: hidden_states = dropout(output) + residual
    │
    ▼
Output hidden_states
```

This is the **Pre-LayerNorm** (Pre-LN) architecture used by GPT-2/3, LLaMA, and most modern LLMs. The layer norm is applied *before* attention/MLP, not after.

### Why Pre-LN?

Post-LN (original Transformer) suffers from training instability at scale. Pre-LN provides more stable gradients because the residual connection passes through without transformation. This is why virtually all modern LLMs use Pre-LN.

---

## 3.6 Self-Attention In Depth

**File**: `megatron/core/transformer/attention.py`

### SelfAttention Class

`SelfAttention` extends the abstract `Attention` class:

```python
class SelfAttention(Attention):
    def __init__(self, config, submodules, layer_number, attn_mask_type, ...):
        # QKV projection: projects hidden_size → (Q + K + V) in one fused GEMM
        self.linear_qkv = ColumnParallelLinear(
            hidden_size,
            query_projection_size + 2 * kv_projection_size,  # Fused QKV
            ...
        )
        
        # Core attention: scaled dot-product attention
        self.core_attention = DotProductAttention(...)
        
        # Output projection: projects attention output back to hidden_size
        self.linear_proj = RowParallelLinear(
            query_projection_size, hidden_size, ...
        )
```

### Fused QKV Projection

Instead of three separate linear layers for Q, K, V, Megatron uses a **single fused linear** that produces all three at once:

```
hidden_states [s, b, h]
      │
      ▼
 linear_qkv: h → (q_size + k_size + v_size)
      │
      ▼
 Split into Q [s, b, nq, d], K [s, b, nkv, d], V [s, b, nkv, d]
```

Where:
- `nq` = number of query heads
- `nkv` = number of key/value heads (can be fewer with GQA)
- `d` = head dimension (`kv_channels`)

This fused approach is more efficient because it's a single GEMM instead of three.

### Grouped-Query Attention (GQA)

Modern LLMs like LLaMA 2/3 use GQA where `num_query_groups < num_attention_heads`. For example, with 32 query heads and 8 KV heads:

```
Q:  32 heads × 128 dims = 4096 dims
K:   8 heads × 128 dims = 1024 dims  (each KV head shared by 4 Q heads)
V:   8 heads × 128 dims = 1024 dims
```

The `Attention` class handles this via:
```python
self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
self.num_query_groups_per_partition = divide(num_query_groups, world_size)
```

### Rotary Position Embeddings (RoPE)

After QKV projection, RoPE is applied to Q and K:

```python
# In SelfAttention.forward():
query, key, value = self.get_query_key_value_tensors(hidden_states)

# Apply rotary embeddings
if rotary_pos_emb is not None:
    query = apply_rotary_pos_emb(query, rotary_pos_emb)
    key = apply_rotary_pos_emb(key, rotary_pos_emb)
```

RoPE encodes relative position information by rotating the Q/K vectors. The rotation angle depends on position, so `dot(Q_i, K_j)` naturally encodes the distance `|i-j|`.

### Core Attention (DotProductAttention)

**File**: `megatron/core/transformer/dot_product_attention.py`

This is the standard scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V
```

Megatron's `DotProductAttention` supports:
- **Fused softmax**: `FusedScaleMaskSoftmax` for efficient masked softmax
- **Query-key scaling**: Optional per-layer scaling for training stability
- **Window attention**: Sliding window patterns for long sequences
- **Attention dropout**: Applied after softmax

> **Note**: In practice, most users will use Transformer Engine's `TEDotProductAttention` which wraps FlashAttention 2/3 for much better performance. The `DotProductAttention` class is the fallback for environments without TE.

---

## 3.7 Multi-Latent Attention (MLA)

**File**: `megatron/core/transformer/multi_latent_attention.py`

MLA is the attention mechanism used by DeepSeek-V2/V3. Instead of storing full KV caches, it compresses KV into a low-rank latent representation:

```
Standard GQA:
  KV cache per layer = 2 × nkv × d × seq_len

MLA:
  Store compressed latent c_kv (much smaller than full KV)
  KV cache per layer = latent_dim × seq_len  (where latent_dim << 2 × nkv × d)
```

The key idea:
1. **Down-project** input → compressed latent `c_kv`
2. **Up-project** `c_kv` → full K and V for attention computation
3. During inference, only store `c_kv` in the KV cache

Megatron supports MLA via `MLASelfAttention` with submodules:
- `linear_q_proj`, `linear_q_down_proj`, `linear_q_up_proj` — Q path with compression
- `linear_kv_down_proj`, `linear_kv_up_proj` — KV path with compression
- `q_layernorm`, `kv_layernorm` — Optional norms after up-projection

---

## 3.8 The MLP (Feed-Forward Network)

**File**: `megatron/core/transformer/mlp.py`

### Standard Dense MLP

```python
class MLP(MegatronModule):
    def __init__(self, config, submodules, ...):
        # Up-projection: h → 4h (or h → 2×4h for gated variants)
        self.linear_fc1 = ColumnParallelLinear(hidden_size, ffn_hidden_size, ...)
        
        # Down-projection: 4h → h
        self.linear_fc2 = RowParallelLinear(ffn_hidden_size, hidden_size, ...)
```

### Activation Functions

The MLP uses gated linear units for modern models:

| Config | Activation | Formula |
|--------|-----------|---------|
| `gated_linear_unit=False` | GeLU | `GeLU(xW₁)W₂` |
| `gated_linear_unit=True, activation_func=silu` | **SwiGLU** | `(SiLU(xW_gate) ⊙ xW_up)W₂` |
| `gated_linear_unit=True, activation_func=gelu` | GeGLU | `(GeLU(xW_gate) ⊙ xW_up)W₂` |

**SwiGLU** (used by LLaMA, Mistral, etc.) splits the `fc1` output into two halves: one goes through SiLU activation, the other is the "gate" — they're multiplied element-wise before the down-projection.

When `gated_linear_unit=True`, `ffn_hidden_size` is doubled internally so that `fc1` outputs `2 × ffn_hidden_size`, which gets split into gate and up projections.

### Tensor Parallelism in MLP

The MLP is split across TP ranks:

```
fc1 = ColumnParallelLinear:  Splits output dimension across TP ranks
                              Each rank computes h → 4h/tp
                              
fc2 = RowParallelLinear:     Splits input dimension across TP ranks
                              Each rank computes 4h/tp → h
                              All-reduce to combine results
```

This is the classic Megatron-LM tensor parallelism pattern where `ColumnParallel` and `RowParallel` are paired to minimize communication (only one all-reduce per MLP).

---

## 3.9 Mixture of Experts (MoE)

**Directory**: `megatron/core/transformer/moe/`

For MoE models (like Mixtral, DeepSeek-V3, Qwen3-MoE), the dense MLP is replaced by a `MoELayer`.

### MoE Architecture

```
MoELayer
├── Router              # Determines which experts process which tokens
│   └── linear(h → num_experts)  # Produces routing scores
├── Experts             # Multiple MLP instances
│   ├── Expert[0]: MLP(h → 4h → h)
│   ├── Expert[1]: MLP(h → 4h → h)
│   └── ...
├── SharedExperts       # (optional) Always-active shared expert
└── TokenDispatcher     # Routes tokens to/from experts across GPUs
```

### Routing

The router produces a score for each token-expert pair, then selects the top-K experts per token:

```python
# Simplified routing
scores = router_linear(hidden_states)          # [tokens, num_experts]
topk_weights, topk_indices = topk(scores, k)   # Select top-K experts
```

### Expert Parallelism (EP)

With EP, different experts live on different GPUs. The `TokenDispatcher` handles all-to-all communication:

```
Tokens on GPU 0: [t1→E2, t2→E0, t3→E1]
Tokens on GPU 1: [t4→E0, t5→E2, t6→E1]

After All-to-All dispatch:
GPU 0 (has E0): processes [t2, t4]
GPU 1 (has E1): processes [t3, t6]
GPU 2 (has E2): processes [t1, t5]

After All-to-All combine:
Results routed back to original GPUs
```

### MoE Layer Frequency

Not every layer needs to be MoE. The `moe_layer_freq` config controls the pattern:

```python
# Integer: every N-th layer is MoE
moe_layer_freq = 2  →  [MoE, Dense, MoE, Dense, ...]

# List: explicit pattern per layer
moe_layer_freq = [1, 0, 1, 0, 1, 1, ...]  →  Custom pattern
```

---

## 3.10 Embeddings and Output Layer

### Token Embeddings

**File**: `megatron/core/models/common/embeddings/language_model_embedding.py`

```python
class LanguageModelEmbedding(MegatronModule):
    def __init__(self, ...):
        self.word_embeddings = VocabParallelEmbedding(vocab_size, hidden_size)
        if position_embedding_type == 'learned_absolute':
            self.position_embeddings = torch.nn.Embedding(max_seq_length, hidden_size)
```

`VocabParallelEmbedding` shards the vocabulary table across TP ranks — each rank only stores `vocab_size / tp_size` rows.

### Positional Encoding Variants

| Type | Description | Models Using It |
|------|-------------|-----------------|
| `learned_absolute` | Learned position embedding table | GPT-2/3 |
| `rope` | Rotary Position Embeddings | LLaMA, Mistral, Qwen |
| `yarn` | YaRN (Yet Another RoPE eNhancement) | Extended context models |
| `mrope` | Multimodal RoPE | Qwen-VL |
| `none` | No positional encoding | Some experimental models |

### Output Layer

The output layer projects from `hidden_size` to `vocab_size`:

```python
self.output_layer = ColumnParallelLinear(
    hidden_size, vocab_size, bias=False, gather_output=not parallel_output
)
```

When `share_embeddings_and_output_weights=True`, the output layer shares weights with the token embedding — this is a common technique that reduces parameters and can improve training stability.

---

## 3.11 TransformerConfig: The Control Center

**File**: `megatron/core/transformer/transformer_config.py`

`TransformerConfig` is a massive dataclass (2000+ lines) that controls every aspect of the model. Here are the key groups:

### Architecture Parameters
```python
num_layers: int                # Number of transformer layers
hidden_size: int               # Model dimension (d_model)
ffn_hidden_size: int           # Feed-forward intermediate size (typically 4×hidden)
num_attention_heads: int       # Number of Q heads
num_query_groups: int          # Number of KV heads (for GQA)
kv_channels: int               # Head dimension
gated_linear_unit: bool        # Use SwiGLU/GeGLU
```

### Precision Parameters
```python
fp16: bool                     # FP16 training
bf16: bool                     # BF16 training
fp8: str                       # FP8 recipe (None, 'hybrid', 'delayed')
fp4: bool                      # FP4 quantization
```

### Regularization
```python
hidden_dropout: float          # Dropout after attention/MLP
attention_dropout: float       # Dropout in attention scores
```

### MoE Parameters
```python
num_moe_experts: int           # Number of experts (None = dense)
moe_layer_freq: int|list       # Which layers are MoE
moe_grouped_gemm: bool         # Use grouped GEMM for efficiency
```

### Recomputation
```python
recompute_granularity: str     # 'full', 'selective', or None
recompute_method: str          # 'uniform' or 'block'
recompute_num_layers: int      # How many layers to checkpoint
recompute_modules: list        # Which modules to selectively recompute
```

---

## 3.12 Putting It All Together: Model Construction Call Chain

```
pretrain_gpt.py :: model_provider()
    │
    ├── get_gpt_decoder_block_spec(config, use_transformer_engine)
    │   ├── get_gpt_layer_with_transformer_engine_spec() or get_gpt_layer_local_spec()
    │   │   └── Returns TransformerLayerSubmodules with all module specs
    │   ├── Handles MoE layer pattern (dense vs. expert layers)
    │   └── Returns TransformerBlockSubmodules(layer_specs=[...], layer_norm=...)
    │
    └── GPTModel(config, transformer_layer_spec, vocab_size, max_seq_len, ...)
        │
        ├── LanguageModelEmbedding(config, vocab_size, max_seq_len)
        │   ├── VocabParallelEmbedding(vocab_size, hidden_size)
        │   └── (optional) PositionEmbedding
        │
        ├── RotaryEmbedding(kv_channels, rotary_percent, ...)  # if RoPE
        │
        ├── TransformerBlock(config, spec)
        │   ├── _build_layers()
        │   │   └── for each layer_spec: build_module(layer_spec, config, layer_number)
        │   │       └── TransformerLayer(config, submodules)
        │   │           ├── input_layernorm
        │   │           ├── SelfAttention(config, submodules)
        │   │           │   ├── linear_qkv (ColumnParallelLinear)
        │   │           │   ├── core_attention (DotProductAttention / TEDotProductAttention)
        │   │           │   └── linear_proj (RowParallelLinear)
        │   │           ├── self_attn_bda
        │   │           ├── pre_mlp_layernorm
        │   │           ├── MLP(config, submodules) or MoELayer(config, ...)
        │   │           │   ├── linear_fc1 (ColumnParallelLinear)
        │   │           │   └── linear_fc2 (RowParallelLinear)
        │   │           └── mlp_bda
        │   └── final_layernorm
        │
        └── output_layer (ColumnParallelLinear → vocab_size)
```

---

## Key Files Covered

| File | Role |
|------|------|
| `megatron/core/models/gpt/gpt_model.py` | Top-level GPT model: embed → decode → output |
| `megatron/core/models/gpt/gpt_layer_specs.py` | Factory functions for layer specs (TE / Local / Inference) |
| `megatron/core/transformer/transformer_block.py` | Layer stack container, activation checkpointing |
| `megatron/core/transformer/transformer_layer.py` | Single transformer layer (LN → Attn → LN → MLP) |
| `megatron/core/transformer/transformer_config.py` | 2000+ line config dataclass |
| `megatron/core/transformer/attention.py` | Attention base class, SelfAttention, CrossAttention |
| `megatron/core/transformer/dot_product_attention.py` | Core scaled dot-product attention |
| `megatron/core/transformer/multi_latent_attention.py` | DeepSeek MLA implementation |
| `megatron/core/transformer/mlp.py` | Dense MLP with SwiGLU/GeLU/GeGLU |
| `megatron/core/transformer/moe/moe_layer.py` | MoE layer with router + experts |
| `megatron/core/transformer/moe/router.py` | Token-to-expert routing |
| `megatron/core/transformer/moe/experts.py` | Expert implementations (Sequential, GroupedGEMM) |
| `megatron/core/transformer/moe/token_dispatcher.py` | All-to-all token routing for EP |
| `megatron/core/transformer/spec_utils.py` | `ModuleSpec` and `build_module` |
| `megatron/core/models/common/embeddings/` | Token embeddings, RoPE, YaRN |

---

## Interview-Ready Takeaways

1. **Pre-LN vs Post-LN**: Modern LLMs use Pre-LayerNorm (norm before attention/MLP) for training stability. Megatron implements this in `TransformerLayer`.

2. **Fused QKV**: Instead of separate Q, K, V linear layers, Megatron uses a single fused `linear_qkv` for efficiency. The output is split into Q, K, V after the GEMM.

3. **Grouped-Query Attention (GQA)**: Reduces KV cache size by sharing KV heads across multiple query heads. Controlled by `num_query_groups` vs `num_attention_heads`.

4. **SwiGLU**: The dominant activation in modern LLMs. Uses gated linear units: `SiLU(xW_gate) ⊙ xW_up`, which requires doubling `ffn_hidden_size` internally.

5. **The Spec Pattern**: Megatron decouples *what* to build from *how* to build it. `ModuleSpec` objects describe the architecture; `BackendSpecProvider` implementations (TE, Local, Inference) provide the actual module classes. This enables FP8 training, custom attention, and MoE without changing model code.

6. **TP in Attention/MLP**: ColumnParallelLinear (splits output dim) paired with RowParallelLinear (splits input dim) minimizes communication to one all-reduce per attention block and one per MLP block.

7. **MoE Routing**: Tokens are dynamically routed to top-K experts. Expert Parallelism distributes experts across GPUs with all-to-all communication for token dispatch.

8. **Multi-Latent Attention (MLA)**: DeepSeek's approach compresses KV into a low-rank latent before up-projecting back, dramatically reducing KV cache size at inference.

9. **Activation Checkpointing**: Three granularities — `full` (checkpoint all layers), `selective` (checkpoint specific ops like attention/layernorm), and none. Trades compute for memory.

10. **Weight Tying**: `share_embeddings_and_output_weights` ties the input embedding and output projection, reducing parameters by `vocab_size × hidden_size`.

---

*Next Chapter: [Chapter 4 — The Pretraining Loop](ch04_pretraining_loop.md) — how Megatron orchestrates forward passes, backward passes, optimizer steps, and logging across distributed workers.*
