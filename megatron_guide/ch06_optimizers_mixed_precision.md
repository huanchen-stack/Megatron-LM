# Chapter 6 — Optimizers, Mixed Precision & Memory Efficiency

> **Goal**: Understand how Megatron-LM manages optimizer state, gradient scaling, mixed-precision training, and activation recomputation — the techniques that make billion-parameter training fit in memory and converge properly.

---

## 6.1 Why This Chapter Matters

Training a 70B-parameter model in BF16 needs ~140 GB just for weights. Adam optimizer doubles that (two extra states per parameter), and FP32 master copies triple it again. Without careful memory management you will OOM long before you see a single loss value.

This chapter covers four pillars:

| Pillar | What It Solves |
|--------|---------------|
| Mixed-precision training | Trains in FP16/BF16 while maintaining FP32 "master" weights for numerical stability |
| Gradient scaling | Prevents FP16 gradients from underflowing to zero |
| Distributed optimizer | Shards optimizer state across DP ranks (ZeRO) |
| Activation recomputation | Trades compute for memory by recomputing activations during backward |

---

## 6.2 The Optimizer Class Hierarchy

Megatron wraps PyTorch's native optimizers (Adam, SGD) in its own hierarchy that adds mixed-precision management, gradient clipping, and distributed sharding.

```
MegatronOptimizer (ABC)                   ← base class
├── FP32Optimizer                         ← simple case: everything in FP32
├── MixedPrecisionOptimizer               ← base for FP16/BF16 training
│   ├── Float16OptimizerWithFloat16Params ← non-distributed mixed precision
│   └── DistributedOptimizer              ← ZeRO-style sharded optimizer
└── ChainedOptimizer                      ← chains multiple optimizers (e.g., dense + MoE)
```

### MegatronOptimizer (base class)

Every optimizer in Megatron inherits from `MegatronOptimizer`, which defines the contract:

```python
class MegatronOptimizer(ABC):
    def __init__(self, optimizer, config, init_state_fn):
        self.optimizer = optimizer   # underlying PyTorch optimizer
        self.config = config         # OptimizerConfig dataclass
    
    # Key abstract methods:
    def prepare_grads(self) -> bool:        # pre-process gradients, return True if inf/nan found
    def step_with_ready_grads(self) -> bool: # call optimizer.step(), copy params back
    def step(self):                          # full step: prepare_grads → clip → step
    def zero_grad(self):                     # clear gradients
    def get_loss_scale(self):                # return current loss scale
    def clip_grad_norm(self, clip_grad):     # compute and clip gradient norm
```

The `step()` method in `MixedPrecisionOptimizer` orchestrates the entire update cycle:

```
step()
 ├── prepare_grads()
 │   ├── copy model grads (FP16/BF16) → main grads (FP32)
 │   ├── unscale gradients (÷ loss_scale)
 │   ├── check for inf/nan
 │   └── update loss scale (DynamicGradScaler)
 ├── clip_grad_norm()          ← global L2 norm clipping
 ├── count_zeros()             ← optional logging
 └── step_with_ready_grads()
     ├── optimizer.step()      ← actual Adam/SGD update on FP32 master params
     └── copy main params (FP32) → model params (FP16/BF16)
```

---

## 6.3 Mixed-Precision Training Deep Dive

### 6.3.1 The Three-Copy Pattern

When training in FP16 or BF16, Megatron maintains three groups of parameters:

| Group | Precision | Purpose |
|-------|-----------|---------|
| `float16_groups` | FP16/BF16 | The actual model weights used in forward/backward |
| `fp32_from_float16_groups` | FP32 | Master copy for optimizer updates (avoids precision loss) |
| `fp32_from_fp32_groups` | FP32 | Parameters that were already FP32 (rare, e.g., layernorm) |

The initialization in `Float16OptimizerWithFloat16Params.__init__()`:

```python
# For each FP16/BF16 parameter:
main_param = param.detach().clone().float()  # FP32 master copy
param.main_param = main_param               # back-reference

# Replace optimizer's parameter with the FP32 copy
param_group['params'][i] = main_param
```

**Why master copies?** Adam computes `m = β₁·m + (1-β₁)·g` and `v = β₂·v + (1-β₂)·g²`. In FP16 (5-bit exponent, 10-bit mantissa), the small update `lr × m / (√v + ε)` can vanish when added to a large weight. FP32 (8-bit exponent, 23-bit mantissa) preserves these updates.

### 6.3.2 Precision Formats in Megatron

Megatron supports several precision formats, controlled via `OptimizerConfig`:

| Format | Flag | Loss Scale | Master Weights | Typical Use |
|--------|------|-----------|----------------|-------------|
| FP32 | neither `fp16` nor `bf16` | None | Not needed | Debugging |
| FP16 | `fp16=True` | Dynamic (required) | FP32 copies | Legacy |
| BF16 | `bf16=True` | None or constant | FP32 copies | **Default for modern training** |
| FP8 (delayed) | `fp8_recipe="delayed"` | Dynamic | FP32 copies | H100/B200 compute |
| MXFP8 | `fp8_recipe="mxfp8"` | Dynamic | FP32 copies | Latest format |

**BF16 vs FP16**: BF16 has the same exponent range as FP32 (8 bits), so it rarely overflows/underflows. FP16 has only 5 exponent bits, requiring dynamic loss scaling. This is why modern training universally prefers BF16.

### 6.3.3 Precision-Aware Optimizer

For extreme memory savings, Megatron supports storing optimizer states in reduced precision:

```python
@dataclass
class OptimizerConfig:
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: torch.dtype = torch.float32    # can be bf16
    main_params_dtype: torch.dtype = torch.float32   # can be bf16
    exp_avg_dtype: torch.dtype = torch.float32       # Adam m: can be bf16
    exp_avg_sq_dtype: torch.dtype = torch.float32    # Adam v: can be bf16
    store_param_remainders: bool = True              # store only the FP32 bits not in BF16
```

The `store_param_remainders` trick is clever: since BF16 shares the same exponent bits as FP32, you only need to store the extra 16 mantissa bits as an `int16`. This halves the master-weight overhead.

---

## 6.4 Gradient Scaling

### 6.4.1 Why Scale Gradients?

FP16 has a tiny range (~6×10⁻⁸ to 65504). Small gradient values that are normal in FP32 become zero in FP16. Loss scaling multiplies the loss by a large factor before backward, which pushes gradients into representable range, then divides them back after backward.

### 6.4.2 DynamicGradScaler

Megatron's `DynamicGradScaler` automatically adjusts the scale:

```python
class DynamicGradScaler(MegatronGradScaler):
    def __init__(self, initial_scale, min_scale, growth_factor, 
                 backoff_factor, growth_interval, hysteresis):
        self._scale = initial_scale   # starts at 2^32
        self.growth_factor = 2.0      # scale up by 2×
        self.backoff_factor = 0.5     # scale down by ½
        self.growth_interval = 1000   # grow after 1000 clean steps
        self.hysteresis = 2           # tolerate 2 consecutive NaN before scaling down
```

**The algorithm:**
1. After each step, check if any gradient was inf/nan
2. If inf/nan found → decrement hysteresis tracker; if tracker reaches 0, multiply scale by `backoff_factor`
3. If no inf/nan → increment growth tracker; after `growth_interval` clean steps, multiply scale by `growth_factor`

The unscaling happens in `_unscale_main_grads_and_check_for_nan()`:

```python
def _unscale_main_grads_and_check_for_nan(self):
    self.found_inf.fill_(0.0)
    # Fused unscale + inf check (one kernel call)
    torch._amp_foreach_non_finite_check_and_unscale_(
        main_grads, self.found_inf, self.grad_scaler.inv_scale
    )
    # Reduce across all model-parallel ranks
    torch.distributed.all_reduce(self.found_inf, op=ReduceOp.MAX, ...)
```

**BF16 note**: Because BF16 has full FP32 exponent range, loss scaling is usually unnecessary. Megatron sets `grad_scaler = None` for BF16 by default.

---

## 6.5 Gradient Clipping

Gradient clipping prevents exploding gradients by capping the global L2 norm:

```python
def clip_grad_norm(self, clip_grad):
    grads_for_norm = self.get_main_grads_for_grad_norm()
    grad_norm = get_grad_norm_fp32(grads_for_norm, group=self.get_grad_stats_parallel_group())
    clip_grad_by_total_norm_fp32(params, clip_grad, grad_norm)
    return grad_norm
```

### Computing Global Grad Norm

The `get_grad_norm_fp32()` function:

1. Computes per-rank local norm using `multi_tensor_l2norm` (fused CUDA kernel from TE/Apex)
2. Squares it: `total_norm = grad_norm ** 2`
3. All-reduces across the appropriate group (model-parallel for non-distributed; entire world for distributed optimizer)
4. Takes square root: `total_norm = total_norm ** 0.5`

### Applying the Clip

```python
clip_coeff = max_norm / (total_norm + 1e-6)
if clip_coeff < 1.0:
    # Scale all gradients by clip_coeff (fused kernel)
    multi_tensor_scale(grads, grads, clip_coeff)
```

The default `clip_grad=1.0` in `OptimizerConfig`.

---

## 6.6 The Distributed Optimizer (ZeRO)

You learned about ZeRO in Chapter 5. Here we dive into Megatron's implementation: `DistributedOptimizer`.

### 6.6.1 Core Idea Recap

Instead of every DP rank storing the full optimizer state, each rank "owns" a 1/N shard:

```
Grad Buffer (padded to be divisible by DP world size):
┌────────────┬────────────┬────────────┬────────────┐
│  DP Rank 0 │  DP Rank 1 │  DP Rank 2 │  DP Rank 3 │
│  (owns)    │  (owns)    │  (owns)    │  (owns)    │
└────────────┴────────────┴────────────┴────────────┘
```

### 6.6.2 Range Maps

The most important data structure in `DistributedOptimizer` is the **range map** — a mapping from each parameter to the slice of the gradient buffer it owns:

```python
param_range_map[param] = {
    "gbuf_world":           Range(...)  # param's range in the full grad buffer
    "gbuf_world_in_bucket": Range(...)  # param's range within its bucket  
    "gbuf_local":           Range(...)  # param's range in this DP rank's view
    "param":                Range(...)  # range within the parameter itself (its shard)
}
```

**Key insight**: Parameter boundaries don't align with DP shard boundaries. A parameter might be split across two DP ranks. Each rank only stores the FP32 master copy and optimizer states for its shard.

### 6.6.3 Shard Construction

During `__init__()`, the distributed optimizer creates sharded views:

```python
# For each FP16/BF16 model parameter:
shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
shard_main_param = shard_model_param.clone().float()  # FP32 shard
model_param.main_param = shard_main_param
```

The inner Adam optimizer operates on these shards, not full parameters.

### 6.6.4 Communication Pattern

| Phase | Operation | Communication |
|-------|-----------|--------------|
| After backward | Reduce-scatter gradients | Each rank gets its 1/N shard of reduced gradients |
| After optimizer.step() | All-gather parameters | Each rank broadcasts its updated shard |

This is handled by the `_ParamAndGradBuffer` (from Chapter 5). The distributed optimizer hooks into the same bucket system.

### 6.6.5 Memory Savings

For a model with P parameters:

| Component | Non-Distributed | Distributed (N ranks) |
|-----------|----------------|----------------------|
| FP32 master weights | 4P bytes | 4P/N bytes |
| Adam exp_avg (m) | 4P bytes | 4P/N bytes |
| Adam exp_avg_sq (v) | 4P bytes | 4P/N bytes |
| **Total optimizer memory** | **12P bytes** | **12P/N bytes** |

For a 70B model with 64 DP ranks: 840 GB → 13.1 GB per rank.

---

## 6.7 Adam, Muon & Optimizer Selection

### 6.7.1 Adam / AdamW

The default optimizer. Megatron uses fused implementations from TransformerEngine or Apex for speed:

```python
try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
except ImportError:
    from apex.optimizers import FusedAdam as Adam       # fallback
except ImportError:
    from torch.optim import AdamW as Adam               # last resort
```

Key Adam config in `OptimizerConfig`:
- `adam_beta1 = 0.9` — first moment decay
- `adam_beta2 = 0.999` — second moment decay  
- `adam_eps = 1e-8` — numerical stability
- `decoupled_weight_decay = True` — use AdamW (weight decay applied to params directly, not through gradient)

### 6.7.2 Muon Optimizer

Muon is a newer optimizer that applies Newton-Schulz orthogonalization to the momentum before updating. It's designed for matrix-shaped parameters (linear layers).

```python
class TensorParallelMuon(OrthogonalizedOptimizer):
    def orthogonalize(self, p, grad):
        orth_grad = newton_schulz_tp(grad, steps=num_ns_steps, ...)
        scale_factor = get_muon_scale_factor(rows, cols, mode=scale_mode)
        return orth_grad * scale_factor
```

Megatron's Muon integration:
1. **Splits parameters**: linear (matrix) params → Muon, non-linear (bias, layernorm, embeddings) → Adam
2. **Chains optimizers**: `ChainedOptimizer([muon_optimizer, adam_optimizer])`
3. **Tensor-parallel aware**: Newton-Schulz iteration accounts for TP-sharded matrices

Key Muon config:
- `muon_momentum = 0.95` — SGD momentum for internal optimizer
- `muon_num_ns_steps = 5` — Newton-Schulz iteration count
- `muon_split_qkv = True` — split QKV for per-head orthogonalization
- `muon_tp_mode = "blockwise"` — how to handle TP-sharded weights

### 6.7.3 ChainedOptimizer

When you have both dense and MoE parameters, or Muon + Adam, Megatron chains them:

```python
class ChainedOptimizer(MegatronOptimizer):
    def __init__(self, chained_optimizers):
        self.chained_optimizers = chained_optimizers  # list of MegatronOptimizer
    
    def step_with_ready_grads(self):
        for optimizer in self.chained_optimizers:
            optimizer.step_with_ready_grads()
```

The `get_megatron_optimizer()` factory function in `__init__.py` automatically creates a `ChainedOptimizer` when there are both regular and expert-parallel parameters.

---

## 6.8 Learning Rate Scheduling

The `OptimizerParamScheduler` controls learning rate and weight decay over training:

### 6.8.1 LR Schedule

```python
class OptimizerParamScheduler:
    def get_lr(self, param_group):
        # Phase 1: Linear warmup
        if self.num_steps <= self.lr_warmup_steps:
            return init_lr + (max_lr - init_lr) * num_steps / lr_warmup_steps
        
        # Phase 2: Decay (after warmup)
        decay_ratio = (num_steps - warmup_steps) / (decay_steps - warmup_steps)
        
        if style == 'cosine':
            coeff = 0.5 * (cos(π * decay_ratio) + 1.0)
        elif style == 'linear':
            coeff = 1.0 - decay_ratio
        elif style == 'WSD':  # warmup-stable-decay
            # Constant LR until wsd_decay_steps before end
            ...
        
        return min_lr + coeff * (max_lr - min_lr)
```

Supported decay styles:
- **`cosine`** — most common, smooth decay following cosine curve
- **`linear`** — simple linear decay
- **`WSD`** (Warmup-Stable-Decay) — constant LR for most of training, then rapid decay
- **`inverse-square-root`** — used in some transformer papers
- **`constant`** — no decay

### 6.8.2 Weight Decay Scheduling

Weight decay can also be scheduled (increasing over training):

```python
def get_wd(self, param_group):
    incr_ratio = num_steps / wd_incr_steps
    if style == 'linear':
        coeff = incr_ratio
    elif style == 'cosine':
        coeff = 0.5 * (cos(π * (1 - incr_ratio)) + 1.0)
    return start_wd + coeff * (end_wd - start_wd)
```

### 6.8.3 Per-Parameter-Group Overrides

Megatron supports different LR/WD for different parameter groups via `ParamKey` matching:

```python
# Skip weight decay for bias and 1D parameters
ParamKey(name="*.bias", predicate=ParamPredicate("param_len_1", fn=lambda p: len(p.shape)==1))
→ ParamGroupOverride(wd_mult=0.0)

# Decoupled LR for embeddings
ParamKey(attr="is_embedding_or_output_parameter")
→ ParamGroupOverride(max_lr=config.decoupled_lr)
```

---

## 6.9 Activation Recomputation (Gradient Checkpointing)

### 6.9.1 The Problem

During forward pass, every layer's activations must be saved for the backward pass. For a 70B model with sequence length 8192, this can require hundreds of GB.

### 6.9.2 The Solution

Instead of storing all activations, **discard** them during forward and **recompute** them during backward:

```
Standard forward:   Layer 1 → save → Layer 2 → save → ... → Layer N → save
Recompute forward:  Layer 1 → discard → Layer 2 → discard → ... → Layer N → discard
Backward:           Recompute Layer K activations on-the-fly when needed
```

**Trade-off**: ~33% more compute (one extra forward pass) in exchange for O(1) activation memory.

### 6.9.3 Granularity Options

Controlled by `TransformerConfig.recompute_granularity`:

| Granularity | What's Recomputed | Memory Saved | Extra Compute |
|-------------|-------------------|--------------|---------------|
| `None` | Nothing | None | None |
| `selective` | Only attention (FlashAttention re-derives softmax) | Moderate | Small |
| `full` | Entire transformer layers | Maximum | ~33% |

### 6.9.4 Method Options (for `full` granularity)

Controlled by `TransformerConfig.recompute_method`:

| Method | Description |
|--------|-------------|
| `uniform` | Divide layers into equal groups; recompute within each group |
| `block` | Recompute the first N layers; store activations for the rest |

With `recompute_num_layers` you control how many layers to recompute.

### 6.9.5 Selective Recomputation

The most practical option. Only the attention core (softmax computation) is recomputed:

```python
# In SelfAttention.forward():
if self.config.recompute_granularity == 'selective':
    # Don't save Q·K^T softmax output — FlashAttention can recompute it
    core_attn_out = self._checkpointed_attention_forward(query, key, value, ...)
```

This saves the O(seq_len²) attention matrix while only adding ~5-10% compute overhead (since FlashAttention's tiling naturally supports recomputation).

---

## 6.10 Putting It All Together: The Optimizer Step

Here's the complete flow when `train_step()` calls `optimizer.step()`:

```
1. Forward pass (FP16/BF16)
   └── Activations saved or discarded (based on recompute config)

2. Loss computation
   └── loss = loss_func(output, labels)
   └── scaled_loss = loss * loss_scale        ← multiply by loss scale

3. Backward pass (FP16/BF16)
   └── Gradients computed in FP16/BF16
   └── Accumulated into gradient buffers

4. optimizer.step()
   ├── prepare_grads()
   │   ├── copy_model_grads_to_main_grads()   ← FP16→FP32 cast
   │   │   (DistOpt: reduce-scatter happens here)
   │   ├── unscale gradients (÷ loss_scale)
   │   ├── check for inf/nan
   │   └── update DynamicGradScaler
   │
   ├── clip_grad_norm()
   │   ├── compute global L2 norm (all-reduce across model-parallel group)
   │   └── scale grads if norm > clip_grad
   │
   ├── step_with_ready_grads()
   │   ├── optimizer.step()                    ← Adam update on FP32 shards
   │   └── copy_main_params_to_model_params()  ← FP32→FP16/BF16 cast
   │       (DistOpt: all-gather happens here)
   │
   └── zero_grad()                             ← clear for next iteration

5. optimizer_param_scheduler.step()
   └── Update LR and WD for all param groups
```

---

## 6.11 Key Files Covered

| File | Purpose |
|------|---------|
| `megatron/core/optimizer/optimizer.py` | `MegatronOptimizer`, `MixedPrecisionOptimizer`, `Float16OptimizerWithFloat16Params`, `FP32Optimizer`, `ChainedOptimizer` |
| `megatron/core/optimizer/distrib_optimizer.py` | `DistributedOptimizer` — ZeRO-style sharded optimizer |
| `megatron/core/optimizer/muon.py` | `TensorParallelMuon`, `get_megatron_muon_optimizer()` |
| `megatron/core/optimizer/clip_grads.py` | `get_grad_norm_fp32()`, `clip_grad_by_total_norm_fp32()` |
| `megatron/core/optimizer/grad_scaler.py` | `MegatronGradScaler`, `DynamicGradScaler`, `ConstantGradScaler` |
| `megatron/core/optimizer/optimizer_config.py` | `OptimizerConfig`, `AdamOptimizerConfig`, `ParamKey` |
| `megatron/core/optimizer/__init__.py` | `get_megatron_optimizer()` factory, param group creation |
| `megatron/core/optimizer_param_scheduler.py` | `OptimizerParamScheduler` — LR and WD scheduling |
| `megatron/core/transformer/transformer_config.py` | `recompute_granularity`, `recompute_method` settings |

---

## 6.12 Interview-Ready Takeaways

1. **Mixed-precision training maintains FP32 master weights** because small Adam updates vanish in FP16. BF16 has the same exponent range as FP32 so it rarely needs loss scaling; FP16 requires dynamic loss scaling.

2. **Dynamic loss scaling** starts high (2³²), backs off by ½ on NaN, grows by 2× after 1000 clean steps. Hysteresis (default 2) prevents thrashing.

3. **Gradient clipping** computes global L2 norm across all model-parallel ranks, then scales all gradients by `min(1, max_norm / total_norm)`. Default `max_norm=1.0`.

4. **DistributedOptimizer (ZeRO)** shards FP32 master weights + Adam states across DP ranks. Communication: reduce-scatter after backward, all-gather after optimizer step. Saves 12P/N bytes per rank.

5. **Activation recomputation** (`selective` vs `full`):
   - `selective`: only recompute attention softmax (~5% overhead, moderate savings)
   - `full`: recompute entire transformer layers (~33% overhead, maximum savings)

6. **Muon optimizer** applies Newton-Schulz orthogonalization to momentum for matrix parameters, while using Adam for non-matrix parameters (bias, LayerNorm, embeddings). It's TP-aware.

7. **Memory budget** for a parameter P in BF16 + distributed Adam:
   - Weight: 2 bytes (BF16)
   - Gradient: 2 bytes (BF16, before reduce-scatter)
   - FP32 master: 4/N bytes (sharded)
   - Adam m: 4/N bytes (sharded)
   - Adam v: 4/N bytes (sharded)
   - Total per rank: 2 + 2 + 12/N bytes per parameter

8. **Cosine LR schedule** is standard: linear warmup → cosine decay to min_lr. WSD (warmup-stable-decay) keeps LR constant for most of training, then decays rapidly — useful when you don't know the total training length upfront.

9. **ChainedOptimizer** handles the common case of different optimizers for dense vs MoE parameters, or Muon vs Adam for linear vs non-linear parameters.

10. **Precision-aware optimizer** can store Adam states in BF16 and use `store_param_remainders=True` to store only the 16 extra mantissa bits of FP32 as int16, halving master-weight memory.
