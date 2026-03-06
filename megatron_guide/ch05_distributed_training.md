# Chapter 5: Distributed Training & Parallelism

## Overview

This chapter covers Megatron's 5D parallelism: Tensor (TP), Pipeline (PP), Data (DP), Context (CP), and Expert (EP) parallelism. Since you're already familiar with these concepts, we'll keep TP/PP/CP/EP concise and focus extra depth on **Data Parallelism math** — gradient all-reduce, reduce-scatter, and ZeRO optimization stages — which are frequently asked in interviews.

---

## 5.1 The 5D Parallelism Hierarchy

Megatron maps GPUs into a 5D mesh:

```
Total GPUs = TP × PP × DP × CP × EP

Example: 512 GPUs
  TP=8, PP=4, DP=8, CP=2, EP=1
  8 × 4 × 8 × 2 × 1 = 512
```

**File**: `megatron/core/parallel_state.py` — Initializes all process groups.

### Process Group Initialization

```python
# Simplified from parallel_state.py
def initialize_model_parallel(
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    context_parallel_size,
    expert_model_parallel_size,
    ...
):
    world_size = torch.distributed.get_world_size()
    data_parallel_size = world_size // (TP * PP * CP * EP)
    
    # Create process groups for each parallelism dimension
    # TP group: GPUs that share a single layer's computation
    # PP group: GPUs forming one pipeline
    # DP group: GPUs that see different data but same model
    # CP group: GPUs that split a long sequence
    # EP group: GPUs that hold different experts
```

---

## 5.2 Tensor Parallelism (TP) — Brief

**What**: Splits individual layers (attention, MLP) across GPUs.

**How in Megatron**: 
- `ColumnParallelLinear`: Splits output dimension. Each GPU computes a column slice.
- `RowParallelLinear`: Splits input dimension. Each GPU computes a partial sum → all-reduce.

```
MLP with TP=2:
GPU 0: fc1 outputs cols [0:2h],  fc2 inputs rows [0:2h]
GPU 1: fc1 outputs cols [2h:4h], fc2 inputs rows [2h:4h]

Communication: One all-reduce after fc2 per layer
```

**Key files**: `megatron/core/tensor_parallel/layers.py`, `mappings.py`

**Sequence Parallelism**: When TP is enabled, LayerNorm and Dropout are parallelized along the sequence dimension to avoid redundant computation. The activation is scattered/gathered at TP boundaries.

---

## 5.3 Pipeline Parallelism (PP) — Brief

**What**: Splits the model depth-wise across GPUs. Each stage holds a subset of layers.

**Schedules** (from `megatron/core/pipeline_parallel/`):

| Schedule | Bubble Ratio | Memory | File |
|----------|-------------|--------|------|
| GPipe (all-forward-all-backward) | `(PP-1)/M` | `M` micro-batches | `schedules.py` |
| 1F1B | `(PP-1)/M` | `PP` micro-batches | `schedules.py` |
| Interleaved 1F1B (Virtual PP) | `(PP-1)/(M×V)` | `PP` micro-batches | `schedules.py` |

Where `M` = num_microbatches, `V` = virtual_pipeline_model_parallel_size.

**Communication**: Point-to-point (send/recv) between adjacent stages. `megatron/core/pipeline_parallel/p2p_communication.py`.

---

## 5.4 Context Parallelism (CP) — Brief

**What**: Splits long sequences across GPUs. Each GPU processes a portion of the sequence.

**How**: For a sequence of length `S` with `CP=K`:
- Each GPU gets `S/K` tokens
- During attention, GPUs communicate KV pairs using ring-based all-to-all

**Key file**: `megatron/core/transformer/dot_product_attention.py` (TEDotProductAttention handles CP internally via `cp_comm_type`)

**Supports**: Both "a2a" (all-to-all) and "p2p" (ring-based) communication patterns.

---

## 5.5 Expert Parallelism (EP) — Brief

**What**: For MoE models, different experts live on different GPUs.

**How**: Tokens are routed to their assigned experts via all-to-all communication.

**Key files**: `megatron/core/transformer/moe/token_dispatcher.py`, `moe_layer.py`

**Communication pattern**:
```
1. Router assigns tokens → experts
2. All-to-All dispatch: send tokens to GPUs that hold their experts
3. Expert computation (local MLP)
4. All-to-All combine: send results back to original GPUs
```

---

## 5.6 Data Parallelism (DP) — In Depth

This is where we go deep, as requested.

### 5.6.1 The Basics

Data parallelism replicates the model on `D` GPUs. Each GPU processes different data, computes gradients, then synchronizes.

```
GPU 0: model_copy_0, data_0 → grad_0
GPU 1: model_copy_1, data_1 → grad_1
GPU 2: model_copy_2, data_2 → grad_2
...
GPU D-1: model_copy_{D-1}, data_{D-1} → grad_{D-1}

Synchronize: avg_grad = (grad_0 + grad_1 + ... + grad_{D-1}) / D
All GPUs: update weights with avg_grad
```

### 5.6.2 Gradient All-Reduce: The Math

**Problem**: Each GPU has gradient `g_i` for parameter `W`. We need the average `ḡ = (1/D) Σ g_i` on all GPUs.

**Naive approach**: Gather all gradients to one GPU, average, broadcast back. Communication: `O(D × |W|)` on the root.

**All-Reduce**: Computes the sum and distributes the result in `O(|W|)` communication per GPU.

#### Ring All-Reduce Algorithm

For `D` GPUs with a parameter of size `N`:

1. **Split** the gradient into `D` chunks of size `N/D`
2. **Reduce-Scatter phase** (`D-1` steps):
   - Each GPU sends one chunk to its right neighbor and receives from left
   - Accumulates (sums) the received chunk with its local chunk
   - After `D-1` steps, each GPU has the complete sum of one chunk
3. **All-Gather phase** (`D-1` steps):
   - Each GPU sends its completed chunk to the right
   - After `D-1` steps, all GPUs have all chunks of the full sum

**Communication volume per GPU**: `2 × (D-1)/D × N ≈ 2N` (for large `D`)

This is **bandwidth-optimal** — each GPU sends and receives exactly `2N` bytes regardless of `D`.

```
Example: 4 GPUs, gradient split into 4 chunks [A, B, C, D]

Initial state:
  GPU 0: [A0, B0, C0, D0]
  GPU 1: [A1, B1, C1, D1]
  GPU 2: [A2, B2, C2, D2]
  GPU 3: [A3, B3, C3, D3]

After Reduce-Scatter (3 steps):
  GPU 0: [_, _, _, D0+D1+D2+D3]  → has chunk D's sum
  GPU 1: [A0+A1+A2+A3, _, _, _]  → has chunk A's sum
  GPU 2: [_, B0+B1+B2+B3, _, _]  → has chunk B's sum
  GPU 3: [_, _, C0+C1+C2+C3, _]  → has chunk C's sum

After All-Gather (3 steps):
  All GPUs: [ΣA, ΣB, ΣC, ΣD]  → complete sum on every GPU
```

To get the **average**, divide by `D` after (or scale each gradient by `1/D` before the all-reduce).

### 5.6.3 Megatron's Gradient Reduction

**File**: `megatron/core/distributed/distributed_data_parallel.py`

Megatron doesn't use PyTorch's built-in DDP. Instead, it manages gradient reduction manually:

```python
class DistributedDataParallel(MegatronModule):
    def __init__(self, config, module, ...):
        # Create contiguous gradient buffers (one per dtype)
        self.grad_buffers = {}  # Maps dtype → GradBuffer
        
    def zero_grad_buffer(self):
        """Zero out gradient buffers (NOT .zero_grad())"""
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.reset()
    
    def start_grad_sync(self):
        """Trigger async gradient all-reduce / reduce-scatter"""
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.start_grad_sync()
```

**Key design choices**:

1. **Contiguous Gradient Buffers**: All gradients for a given dtype are packed into a single contiguous tensor. This enables efficient NCCL all-reduce (fewer kernel launches, better bandwidth utilization).

2. **Bucketed Gradient Reduction**: Parameters are grouped into buckets. Each bucket is reduced as soon as all its gradients are computed (during backward pass), overlapping communication with computation.

3. **Reduce-Scatter vs All-Reduce**: 
   - Standard DDP: All-reduce gradients → every GPU has the full averaged gradient
   - Distributed Optimizer: Reduce-scatter gradients → each GPU only gets its shard of the averaged gradient (saves memory, see ZeRO below)

### 5.6.4 Overlapping Communication with Computation

Megatron overlaps gradient reduction with the backward pass:

```
Backward pass timeline:
  Layer N   backward → gradients for layer N ready → start reduce for bucket containing layer N
  Layer N-1 backward → gradients for layer N-1 ready → start reduce for next bucket
  ...
  Layer 1   backward → last gradients ready → start reduce for last bucket
  
  Meanwhile: NCCL reduce operations from earlier buckets are completing in parallel
```

This is controlled by `--overlap-grad-reduce`. The reduction happens asynchronously via NCCL streams, so it doesn't block the backward computation.

Similarly, `--overlap-param-gather` overlaps parameter all-gather (needed by distributed optimizer) with the forward pass of the next iteration.

---

## 5.7 ZeRO Optimization: The Math

ZeRO (Zero Redundancy Optimizer) is a family of memory optimizations for data parallelism. Megatron implements ZeRO Stage 1+2 via its `DistributedOptimizer`.

### 5.7.1 Memory Analysis Without ZeRO

For a model with `Φ` parameters in mixed precision (FP16 model, FP32 optimizer):

| Component | Memory per GPU | Notes |
|-----------|---------------|-------|
| FP16 Parameters | `2Φ` bytes | Model weights |
| FP16 Gradients | `2Φ` bytes | Gradient buffers |
| FP32 Master Weights | `4Φ` bytes | For accurate accumulation |
| FP32 Momentum (Adam) | `4Φ` bytes | First moment `m` |
| FP32 Variance (Adam) | `4Φ` bytes | Second moment `v` |
| **Total** | **`16Φ` bytes** | Per GPU, redundant across DP |

For a 7B parameter model: `16 × 7B × 1 byte = 112 GB` per GPU. That's **redundant on every DP GPU**.

### 5.7.2 ZeRO Stages

ZeRO partitions the redundant state across `D` data-parallel GPUs:

| Stage | What's Partitioned | Memory per GPU | Communication |
|-------|-------------------|----------------|---------------|
| **Stage 0** | Nothing (standard DDP) | `16Φ` | All-reduce gradients: `2Φ` per GPU |
| **Stage 1** | Optimizer states (`m`, `v`) | `4Φ + 12Φ/D` | All-reduce gradients: `2Φ` per GPU |
| **Stage 2** | Optimizer states + Gradients | `4Φ + 12Φ/D` | Reduce-scatter grads + All-gather params: `2Φ` per GPU |
| **Stage 3** | Everything (params too) | `16Φ/D` | More communication |

### 5.7.3 ZeRO Stage 1: Partition Optimizer States

Each GPU only stores optimizer states for `Φ/D` parameters:

```
GPU 0: owns optimizer state for params [0, Φ/D)
GPU 1: owns optimizer state for params [Φ/D, 2Φ/D)
...
GPU D-1: owns optimizer state for params [(D-1)Φ/D, Φ)
```

After gradient all-reduce, each GPU:
1. Has the full averaged gradient
2. Updates only its owned slice of parameters
3. All-gathers the updated parameters

**Memory**: Each GPU stores `2Φ` (FP16 params) + `2Φ` (FP16 grads) + `12Φ/D` (FP32 master weights + optimizer states for its shard) = `4Φ + 12Φ/D`

### 5.7.4 ZeRO Stage 2: Partition Gradients Too

Instead of all-reducing gradients (every GPU gets full gradient), use **reduce-scatter** (each GPU gets only its shard):

```
Reduce-Scatter:
  Input:  GPU_i has full gradient g_i (size Φ)
  Output: GPU_i has averaged gradient shard ḡ[iΦ/D : (i+1)Φ/D]
  Communication: Each GPU sends/receives Φ(D-1)/D ≈ Φ bytes
```

Then each GPU:
1. Has only its shard of the averaged gradient
2. Updates only its shard's parameters using its shard's optimizer states
3. All-gathers the updated FP16 parameters

**Communication volume**: `Φ` (reduce-scatter) + `Φ` (all-gather) = `2Φ` per GPU — **same as standard all-reduce**!

So ZeRO Stage 2 saves memory without increasing communication. This is why Megatron's `DistributedOptimizer` is the default.

### 5.7.5 Megatron's DistributedOptimizer

**File**: `megatron/core/optimizer/distrib_optimizer.py`

```python
class DistributedOptimizer(MixedPrecisionOptimizer):
    """Distributed optimizer that partitions optimizer state across DP ranks.
    
    Implements ZeRO Stage 1+2:
    - Gradient reduce-scatter (Stage 2): each rank gets its gradient shard
    - Optimizer state partitioning (Stage 1): each rank stores states for its shard
    - Parameter all-gather: broadcast updated params to all ranks
    """
    
    def step(self):
        # 1. Reduce-scatter gradients across DP ranks
        #    Each rank now has the averaged gradient for its parameter shard
        self._reduce_scatter_grads()
        
        # 2. Copy FP16 gradient shard → FP32 (for master weight update)
        self._copy_grads_to_main()
        
        # 3. Gradient clipping (requires global norm across all ranks)
        grad_norm = self._clip_gradients()
        
        # 4. Adam step on FP32 master weight shard
        self.optimizer.step()  # Only updates Φ/D parameters
        
        # 5. Copy updated FP32 shard → FP16
        self._copy_main_to_model()
        
        # 6. All-gather updated FP16 params across DP ranks
        self._all_gather_params()
```

### 5.7.6 Gradient Clipping with Distributed Gradients

A subtlety: gradient clipping requires the **global gradient norm** (across all parameters), but each GPU only has a shard.

```python
# Each GPU computes partial norm for its shard
local_norm_sq = sum(p.grad.norm()**2 for p in my_shard_params)

# All-reduce to get global norm
global_norm_sq = all_reduce(local_norm_sq, op=SUM)
global_norm = sqrt(global_norm_sq)

# Clip
clip_coeff = max_norm / max(global_norm, max_norm)
for p in my_shard_params:
    p.grad *= clip_coeff
```

---

## 5.8 Communication Patterns Summary

| Parallelism | Communication | Frequency | Volume per GPU |
|-------------|--------------|-----------|----------------|
| **TP** | All-reduce / Reduce-scatter | Per layer, per micro-batch | `O(batch × seq × hidden)` |
| **PP** | Point-to-point (send/recv) | Per micro-batch, between stages | `O(batch × seq × hidden)` |
| **DP** | All-reduce or Reduce-scatter + All-gather | Per iteration (gradient sync) | `O(Φ)` model parameters |
| **CP** | Ring all-to-all (KV exchange) | Per attention layer | `O(batch × seq × kv_dim)` |
| **EP** | All-to-all (token dispatch/combine) | Per MoE layer | `O(tokens × hidden)` |

### Communication Overlap

Megatron overlaps as much communication as possible:

```
--overlap-grad-reduce     : DP gradient reduce during backward pass
--overlap-param-gather    : DP param all-gather during forward pass  
--tp-comm-overlap         : TP communication during GEMM computation
```

---

## 5.9 Process Group Organization in Code

```python
# Key functions in megatron/core/parallel_state.py
get_tensor_model_parallel_group()       # TP group
get_pipeline_model_parallel_group()     # PP group
get_data_parallel_group()               # DP group (with_context_parallel=False)
get_data_parallel_group(with_context_parallel=True)  # DP × CP group
get_context_parallel_group()            # CP group
get_expert_model_parallel_group()       # EP group
```

### GPU Mapping Example

For 16 GPUs with TP=2, PP=2, DP=4:

```
GPU  0: TP_group=[0,1],   PP_group=[0,2],   DP_group=[0,4,8,12]
GPU  1: TP_group=[0,1],   PP_group=[1,3],   DP_group=[1,5,9,13]
GPU  2: TP_group=[2,3],   PP_group=[0,2],   DP_group=[2,6,10,14]
GPU  3: TP_group=[2,3],   PP_group=[1,3],   DP_group=[3,7,11,15]
GPU  4: TP_group=[4,5],   PP_group=[4,6],   DP_group=[0,4,8,12]
...
```

GPUs within a TP group are always on the same node (fast NVLink). PP and DP groups may span nodes.

---

## Key Files Covered

| File | Role |
|------|------|
| `megatron/core/parallel_state.py` | Process group initialization for all 5 dimensions |
| `megatron/core/tensor_parallel/layers.py` | ColumnParallelLinear, RowParallelLinear |
| `megatron/core/tensor_parallel/mappings.py` | All-reduce, reduce-scatter, all-gather primitives |
| `megatron/core/pipeline_parallel/schedules.py` | 1F1B, interleaved pipeline schedules |
| `megatron/core/pipeline_parallel/p2p_communication.py` | Send/recv between pipeline stages |
| `megatron/core/distributed/distributed_data_parallel.py` | Custom DDP with gradient buffers |
| `megatron/core/distributed/grad_buffer.py` | Contiguous gradient buffer management |
| `megatron/core/distributed/param_and_grad_buffer.py` | Combined param+grad buffers |
| `megatron/core/optimizer/distrib_optimizer.py` | ZeRO Stage 1+2 distributed optimizer |
| `megatron/core/transformer/moe/token_dispatcher.py` | All-to-all token routing for EP |

---

## Interview-Ready Takeaways

1. **Ring All-Reduce Math**: Communication volume per GPU is `2N(D-1)/D ≈ 2N` bytes for `N` parameter bytes across `D` GPUs. This is bandwidth-optimal — independent of `D`.

2. **Reduce-Scatter vs All-Reduce**: All-reduce gives every GPU the full result (volume: `2N`). Reduce-scatter gives each GPU only its shard (volume: `N`). When combined with all-gather later (another `N`), total is the same `2N`, but allows memory savings between the two operations.

3. **ZeRO Stage 2 is Free**: Reduce-scatter + all-gather has the same communication volume as all-reduce (`2N`), but saves `2Φ` bytes of gradient memory per GPU. This is why it's Megatron's default.

4. **Memory with ZeRO**: Without ZeRO, `16Φ` bytes per GPU. With ZeRO Stage 2 and large `D`: approaches `4Φ + 12Φ/D ≈ 4Φ` bytes per GPU (the FP16 params + grads that must be present for forward/backward).

5. **TP Placement**: TP groups should be within a single node (NVLink bandwidth: ~900 GB/s on H100). PP and DP can span nodes (InfiniBand: ~400 GB/s per port).

6. **Communication Overlap**: Megatron overlaps gradient reduce with backward pass, parameter gather with forward pass, and TP communication with GEMM computation. This hides communication latency.

7. **5D Mapping**: `Total GPUs = TP × PP × DP × CP × EP`. DP is the implicit dimension — it's whatever's left after allocating TP, PP, CP, EP.

8. **Pipeline Bubble**: 1F1B bubble ratio = `(PP-1)/M` where M is num_microbatches. Virtual PP reduces this to `(PP-1)/(M×V)`. To minimize bubbles, you need `M >> PP`.

9. **Gradient Clipping with ZeRO**: Each GPU computes local norm² for its gradient shard, then all-reduce(SUM) to get global norm². This is a single scalar all-reduce — negligible overhead.

10. **Sequence Parallelism**: Complementary to TP. When tensors are not in TP regions (LayerNorm, Dropout), they're split along the sequence dimension instead. This avoids redundant computation and memory for activations.

---

*Next Chapter: [Chapter 6 — Optimizers & Mixed Precision](ch06_optimizers_mixed_precision.md) — the distributed optimizer internals, Muon optimizer, gradient clipping, FP16/BF16/FP8/FP4, and activation recomputation.*
