# Chapter 7 — Checkpointing & Model Conversion

> **Goal**: Understand how Megatron saves and loads distributed checkpoints, how checkpoint resharding works when changing parallelism, and how to convert between Megatron and Hugging Face formats.

---

## 7.1 Why Checkpointing Is Harder Than You Think

When training a 70B model across 512 GPUs with tensor, pipeline, and expert parallelism, the model's state is scattered across every rank. Saving a checkpoint means:

1. Each rank has only a **shard** of the weights (from TP, PP, EP)
2. The optimizer state is further sharded by the distributed optimizer (ZeRO)
3. RNG states differ per rank (for reproducibility)
4. You want to **resume with different parallelism** (e.g., trained on 512 GPUs, fine-tune on 64)
5. Checkpoints must be fast — a 70B model checkpoint can be 200+ GB

Megatron solves this with **distributed checkpointing** (`dist_checkpointing`), built on top of PyTorch's `torch.distributed.checkpoint` (DCP).

---

## 7.2 Checkpoint Types

Megatron supports several checkpoint formats:

| Type | Format | Reshardable? | Description |
|------|--------|-------------|-------------|
| Legacy | Single `.pt` per rank | No | Original Megatron format, one file per TP×PP rank |
| `torch_dist` | PyTorch DCP | Yes | Default modern format, supports resharding |
| `fsdp_dtensor` | DTensor-based | Yes | For Megatron-FSDP workflows |
| Local | Rank-local storage | No | Non-persistent checkpoints on local SSD/ramdisk |

The format is controlled by `--ckpt-format` (default: `torch_dist`).

---

## 7.3 The Distributed Checkpointing Library

Located in `megatron/core/dist_checkpointing/`, this library provides the core abstraction: **describe what each rank owns, and let the system figure out how to save/load it.**

### 7.3.1 ShardedTensor — The Core Abstraction

Every tensor in a distributed model is wrapped in a `ShardedTensor` that describes:

```python
@dataclass
class ShardedTensor:
    key: str                           # unique global name, e.g. "model.layers.0.attn.qkv.weight"
    data: Optional[torch.Tensor]       # the local shard
    dtype: torch.dtype                 # tensor dtype
    local_shape: Tuple[int, ...]       # shape of the local shard
    global_shape: Tuple[int, ...]      # shape of the full (unsharded) tensor
    global_offset: Tuple[int, ...]     # where this shard sits in the global tensor
    axis_fragmentations: Tuple[int, ...] # how many pieces along each axis
    replica_id: ReplicaId              # identifies replicas (DP replicas share same data)
```

**Example**: A QKV weight with TP=4, shape `[3*H/4, H]` on rank 2:
```python
ShardedTensor(
    key="decoder.layers.0.self_attention.linear_qkv.weight",
    data=local_qkv_weight,          # shape [3*H/4, H]
    global_shape=(3*H, H),          # full tensor shape
    global_offset=(2 * 3*H/4, 0),   # this rank's offset
    axis_fragmentations=(4, 1),      # split along dim 0 into 4 pieces
    replica_id=dp_rank,              # DP replicas have same content
)
```

### 7.3.2 ShardedObject

For non-tensor state (RNG states, metadata):

```python
@dataclass
class ShardedObject:
    key: str
    data: object              # arbitrary Python object
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    replica_id: ReplicaId
```

### 7.3.3 Building the Sharded State Dict

Each model and optimizer produces a sharded state dict via `sharded_state_dict()`:

```python
# In GPTModel:
def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
    # Each layer knows its own parallelism and produces ShardedTensors
    sharded_sd = {}
    for layer in self.layers:
        sharded_sd.update(layer.sharded_state_dict(prefix=f"{prefix}layers.{i}.", ...))
    return sharded_sd
```

The key insight: **models describe their sharding, not the checkpoint format**. The checkpoint strategy then handles the actual I/O.

---

## 7.4 Save Path

### 7.4.1 The `save_checkpoint()` Function

Located in `megatron/training/checkpointing.py`, this is the main entry point:

```python
def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, ...):
    # 1. Build state dict
    state_dict = {
        'args': args,
        'checkpoint_version': 3.0,
        'iteration': iteration,
        'tokens': args.consumed_train_tokens,
        'num_floating_point_operations_so_far': ...,
    }
    
    # 2. Get model's sharded state dict
    model_state_dict = model.sharded_state_dict()
    
    # 3. Get optimizer's sharded state dict  
    optimizer_state_dict = optimizer.sharded_state_dict(model_state_dict)
    
    # 4. Get RNG state (different per rank)
    rng_state = get_rng_state(ckpt_format, tp_group, pp_group)
    
    # 5. Save everything
    dist_checkpointing.save(state_dict, checkpoint_dir)
```

### 7.4.2 Save Strategies

The actual I/O is handled by configurable strategies:

| Strategy | Description |
|----------|-------------|
| `torch_dist` | Uses PyTorch DCP's `torch.distributed.checkpoint.save` |
| Fully Parallel | Wraps another strategy with parallelized I/O across ranks |
| Async | Non-blocking save that runs in background |

**Async saving** (`--async-save`): Copies tensors to CPU, then saves in a background thread while training continues. This overlaps I/O with compute.

### 7.4.3 Checkpoint Directory Structure

```
checkpoints/
├── latest_checkpointed_iteration.txt   ← tracker file ("1000")
├── iter_0001000/
│   ├── __0_0.distcp/                   ← PyTorch DCP format
│   │   ├── __0_0.distcp                ← tensor data
│   │   ├── .metadata                   ← shard metadata
│   │   └── ...
│   ├── common.pt                       ← non-sharded state (args, iteration, etc.)
│   └── rng_state/                      ← RNG states
└── iter_0002000/
    └── ...
```

---

## 7.5 Load Path

### 7.5.1 The `load_checkpoint()` Function

```python
def load_checkpoint(model, optimizer, opt_param_scheduler, ...):
    # 1. Find latest checkpoint
    iteration, release = read_metadata(tracker_filename)
    
    # 2. Build the *current* model's sharded state dict (as a template)
    model_state_dict = model.sharded_state_dict()
    
    # 3. Load: the library matches saved shards to requested shards
    loaded_state = dist_checkpointing.load(model_state_dict, checkpoint_dir)
    
    # 4. Apply loaded state
    model.load_state_dict(loaded_state)
```

### 7.5.2 The Resharding Magic

When loading with different parallelism than what was saved, the distributed checkpointing library handles resharding automatically:

**Saved with TP=4, Loading with TP=2:**
```
Saved:   [shard0 | shard1 | shard2 | shard3]    (4 pieces)
Loading: [shard0+shard1 | shard2+shard3]         (2 pieces, each rank reads 2 saved shards)
```

This works because:
1. Save records the `global_shape` and each shard's `global_offset`
2. Load describes the *new* sharding via `ShardedTensor`
3. PyTorch DCP figures out which saved shards overlap with each requested shard
4. Each rank reads only the data it needs

**This is why `torch_dist` format is preferred**: legacy `.pt` format requires the same TP×PP layout for loading.

### 7.5.3 Strict Loading

The `strict` parameter controls mismatch handling:

| Value | Behavior |
|-------|----------|
| `ASSUME_OK_UNEXPECTED` (default) | Ignore unexpected keys in checkpoint |
| `LOG_UNEXPECTED` | Log unexpected keys |
| `RETURN_ALL` | Return missing and unexpected key sets |
| `RAISE_ALL` | Raise error on any mismatch |

---

## 7.6 Optimizer Checkpoint

The distributed optimizer's checkpoint is particularly complex because it involves:

1. **Optimizer state (non-parameter)**: step count, learning rate schedule → saved as common state
2. **Parameter state**: FP32 master weights, Adam m/v → sharded across DP ranks

### 7.6.1 Sharding Formats

| Format | Key | Description |
|--------|-----|-------------|
| `dp_reshardable` | Internal bucket layout | Fast but must match DP degree |
| `fully_reshardable` | Model-space layout | Slower but fully reshardable |
| `fully_sharded_model_space` | Model-space + deduplication | Most flexible |
| `dp_zero_gather_scatter` | Legacy gather/scatter | Compatibility |

The format is selected via `--dist-ckpt-optim-fully-reshardable`.

### 7.6.2 Grad Scaler State

When using FP16 with dynamic loss scaling, the scaler's state is saved:
```python
state_dict['grad_scaler'] = {
    'scale': self._scale,              # current loss scale
    'growth_tracker': self._growth_tracker,
    'hysteresis_tracker': self._hysteresis_tracker,
}
```

---

## 7.7 Checkpoint Conversion (Megatron ↔ HuggingFace)

### 7.7.1 The Convert Tool

`tools/checkpoint/convert.py` provides a plugin-based converter:

```bash
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader legacy \          # or "core" for Megatron-Core format
    --saver hf \               # save to HuggingFace format
    --load-dir /path/to/megatron/checkpoint \
    --save-dir /path/to/hf/output
```

### 7.7.2 Architecture: Loader → Queue → Saver

The conversion uses a multiprocessing queue to decouple loading and saving:

```
┌──────────┐     ┌───────┐     ┌──────────┐
│  Loader  │────→│ Queue │────→│  Saver   │
│ Process  │     │(tensors)    │ Process  │
└──────────┘     └───────┘     └──────────┘
```

The protocol sends tensors in a specific order:
1. **Metadata** (num_layers, hidden_size, etc.)
2. **Embeddings** (word embeddings, position embeddings)
3. **Transformer layers** (one by one: QKV, dense, MLP, norms)
4. **Final layer norm**
5. **"done"** sentinel

### 7.7.3 Available Loaders and Savers

| Name | Module | Direction |
|------|--------|-----------|
| `legacy` | `loader_legacy.py` / `saver_legacy.py` | Megatron legacy format |
| `core` | `loader_core.py` / `saver_core.py` | Megatron-Core format |
| `llama_mistral` | `loader_llama_mistral.py` | HF LLaMA/Mistral → Megatron |
| `mixtral_hf` | `loader_mixtral_hf.py` | HF Mixtral → Megatron |

### 7.7.4 Megatron Bridge

For production-grade conversion, NVIDIA provides **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** — a separate tool with:
- Bidirectional HF ↔ Megatron conversion
- Production-ready recipes for popular models (LLaMA, Mistral, etc.)
- Proper handling of tied weights, RoPE scaling, GQA reshaping

---

## 7.8 Practical Checkpoint Operations

### 7.8.1 Resuming Training

```bash
# Megatron automatically finds the latest checkpoint
python pretrain_gpt.py \
    --load /path/to/checkpoints \      # directory with latest_checkpointed_iteration.txt
    --save /path/to/checkpoints \
    --use-dist-ckpt \
    --ckpt-format torch_dist
```

### 7.8.2 Changing Parallelism

```bash
# Trained with TP=8, PP=4. Fine-tune with TP=4, PP=2 (different GPU count)
python pretrain_gpt.py \
    --load /path/to/tp8_pp4_checkpoint \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 2 \
    --use-dist-ckpt \
    --ckpt-format torch_dist
# Resharding happens automatically during load!
```

### 7.8.3 Non-Persistent Checkpoints

For fault tolerance without filling up storage:

```bash
--non-persistent-ckpt-type global \    # save to shared filesystem
--non-persistent-save-interval 100 \   # save every 100 steps
--save-interval 1000                   # "real" checkpoints every 1000 steps
```

Non-persistent checkpoints automatically delete the previous one when a new one is saved.

### 7.8.4 Async Checkpointing

```bash
--async-save   # save checkpoints in the background
```

The tensors are copied to CPU memory, and the I/O happens in a background thread while training continues on GPU.

---

## 7.9 Checkpoint Inspection

The `tools/checkpoint/checkpoint_inspector.py` allows examining checkpoint contents without loading the full model:

```bash
python tools/checkpoint/checkpoint_inspector.py \
    --ckpt-dir /path/to/checkpoint \
    --ckpt-format torch_dist
```

---

## 7.10 Key Files Covered

| File | Purpose |
|------|---------|
| `megatron/training/checkpointing.py` | `save_checkpoint()`, `load_checkpoint()` — main entry points |
| `megatron/core/dist_checkpointing/__init__.py` | Public API: `save()`, `load()` |
| `megatron/core/dist_checkpointing/mapping.py` | `ShardedTensor`, `ShardedObject` — core abstractions |
| `megatron/core/dist_checkpointing/serialization.py` | `save()`, `load()` implementation with strategy dispatch |
| `megatron/core/dist_checkpointing/core.py` | `CheckpointingConfig`, format detection |
| `megatron/core/dist_checkpointing/strategies/` | I/O strategies (torch_dist, fully_parallel, async) |
| `megatron/core/dist_checkpointing/optimizer.py` | Optimizer state sharding helpers |
| `tools/checkpoint/convert.py` | Checkpoint converter (loader→queue→saver architecture) |
| `tools/checkpoint/loader_*.py` | Format-specific loaders |
| `tools/checkpoint/saver_*.py` | Format-specific savers |

---

## 7.11 Interview-Ready Takeaways

1. **Distributed checkpointing** wraps tensors in `ShardedTensor` descriptors that record each rank's shard of a global tensor. The checkpoint library uses this metadata to save and load without requiring all ranks to communicate.

2. **Resharding** (changing TP/PP/DP between save and load) works because `ShardedTensor` records `global_shape` and `global_offset`. PyTorch DCP computes the overlap between saved and requested shards at load time.

3. **Checkpoint format**: `torch_dist` (PyTorch Distributed Checkpoint) is the default and recommended format. Legacy format saves one `.pt` file per rank and doesn't support resharding.

4. **Optimizer checkpointing** is the hardest part — the distributed optimizer's state is sharded across DP ranks in bucket order, not model parameter order. The `fully_reshardable` format converts to model-space ordering for full reshardability.

5. **Async checkpointing** copies tensors to CPU, then writes to disk in a background thread. Training continues on GPU immediately, overlapping I/O with compute.

6. **The conversion pipeline** (Megatron ↔ HF) uses a loader→queue→saver architecture. Tensors are sent through a multiprocessing queue in a protocol that includes metadata, embeddings, each transformer layer's weights, and final norms.

7. **Megatron Bridge** (separate repo) is the production-grade tool for HF↔Megatron conversion with recipes for popular models.

8. **RNG state** is saved per-rank (different TP and PP ranks have different RNG states) to ensure bit-exact reproducibility on resume.

9. **Non-persistent checkpoints** are for fault tolerance: they're saved frequently but automatically deleted when a newer one is saved, preventing storage bloat.

10. **Checkpoint versioning**: Megatron uses a version number (currently 3.0) and validates model architecture parameters (num_layers, hidden_size, num_attention_heads) match between checkpoint and current config.
