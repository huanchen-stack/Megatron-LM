# Chapter 4: The Pretraining Loop

## Overview

Chapters 1–3 covered data preparation and model architecture. Now we assemble them into a working training system. This chapter traces the entire training loop — from the `pretrain()` entry point through the inner `train_step()` that runs forward passes, backward passes, and optimizer updates.

The key insight is Megatron's **separation of concerns**: the entry script (`pretrain_gpt.py`) defines *what* to train (model, data, loss), while the training engine (`megatron/training/training.py`, ~3600 lines) defines *how* to train (distributed setup, pipeline schedules, gradient handling, checkpointing, logging).

---

## 4.1 The Entry Point: `pretrain_gpt.py`

The entry script is surprisingly small (~350 lines). It provides four callbacks to the training engine:

```python
# pretrain_gpt.py :: __main__
pretrain(
    train_valid_test_datasets_provider,  # How to build datasets
    model_provider,                       # How to build the model
    ModelType.encoder_or_decoder,         # Model type enum
    forward_step,                         # How to run one forward pass
)
```

### The Four Callbacks

| Callback | Purpose | Returns |
|----------|---------|---------|
| `model_provider()` | Instantiates `GPTModel` with the right specs | `GPTModel` instance |
| `train_valid_test_datasets_provider()` | Builds train/valid/test datasets | Three dataset objects |
| `forward_step(data_iterator, model)` | Runs one forward pass on a micro-batch | `(loss_tensor, loss_func)` |
| `loss_func(loss_mask, output_tensor)` | Computes scalar loss from model output | `(loss, num_tokens, report)` |

This design means you can pretrain different model types (GPT, BERT, T5, multimodal) by swapping these callbacks — the training loop stays the same.

---

## 4.2 The `forward_step` Function

This is where a single micro-batch gets processed:

```python
def forward_step(data_iterator, model):
    # 1. Get a batch from the data iterator
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator)
    
    # 2. Run the model forward pass
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, 
                          loss_mask=loss_mask, packed_seq_params=packed_seq_params)
    
    # 3. Return output + loss function (partially applied with loss_mask)
    return output_tensor, partial(loss_func, loss_mask, model=model)
```

### The Loss Function

```python
def loss_func(loss_mask, output_tensor, model=None):
    losses = output_tensor.view(-1).float()      # Per-token losses
    loss_mask = loss_mask.view(-1).float()         # 1.0 for real tokens, 0.0 for padding
    loss = torch.sum(losses * loss_mask)           # Masked sum
    num_tokens = loss_mask.sum()                   # Count of real tokens
    
    return loss, num_tokens, {'lm loss': [loss, num_tokens]}
```

Key detail: the loss is a **sum** (not mean) over tokens in the micro-batch. The mean is computed later by dividing by the total number of tokens across all micro-batches and data-parallel ranks. This ensures mathematically correct loss averaging regardless of padding.

---

## 4.3 The `pretrain()` Function: Orchestrating Everything

**File**: `megatron/training/training.py`, line 718

`pretrain()` is the master orchestrator. It runs these steps in order:

```
pretrain()
│
├── 1. initialize_megatron()          # Parse args, init distributed, set seeds
│   ├── torch.distributed.init_process_group()
│   ├── Initialize model-parallel groups (TP, PP, DP, CP, EP)
│   └── Set random seeds per rank
│
├── 2. setup_model_and_optimizer()    # Build model, optimizer, scheduler
│   ├── model_provider()              # User callback → GPTModel
│   ├── Wrap model in Float16Module   # FP16/BF16 wrapper
│   ├── Wrap model in DDP/FSDP        # Distributed data parallel
│   ├── get_megatron_optimizer()      # Create distributed optimizer
│   ├── OptimizerParamScheduler()     # Learning rate schedule
│   └── load_checkpoint()             # Resume from checkpoint (if any)
│
├── 3. Build datasets and data loaders
│   ├── train_valid_test_datasets_provider()
│   └── build_pretraining_data_loader()
│
├── 4. train()                        # The actual training loop
│   └── while iteration < train_iters:
│       ├── train_step()              # Forward + Backward + Optimizer
│       ├── training_log()            # Log metrics
│       ├── evaluate_and_print_results()  # Periodic validation
│       └── save_checkpoint()         # Periodic checkpointing
│
└── 5. Cleanup
```

---

## 4.4 Model Setup: `setup_model_and_optimizer()`

This function (around line 1000 in training.py) builds and wraps the model:

### Step 1: Build the Raw Model

```python
model = model_provider()  # Returns GPTModel (on CPU or meta device)
```

### Step 2: FP16/BF16 Wrapping

If using mixed precision, the model is wrapped in `Float16Module`:

```python
if config.fp16 or config.bf16:
    model = Float16Module(config, model)
```

`Float16Module` handles:
- Casting inputs to FP16/BF16 before forward pass
- Keeping a FP32 copy of weights for gradient accumulation
- Loss scaling for FP16 training

### Step 3: DDP or FSDP Wrapping

For data parallelism, the model is wrapped in either:

```python
# Standard Megatron DDP
model = DDP(
    config,
    ddp_config,
    model,
    data_parallel_group=dp_group,
    expert_data_parallel_group=ep_group,
)

# OR PyTorch FSDP2 (newer option)
model = TorchFullyShardedDataParallel(model, ...)
```

Megatron's `DDP` is custom — it manages gradient buffers, overlaps gradient reduce with computation, and supports the distributed optimizer. This is NOT `torch.nn.parallel.DistributedDataParallel`.

### Step 4: Create Optimizer

```python
optimizer = get_megatron_optimizer(config, model_chunks, ...)
```

This creates Megatron's `DistributedOptimizer` which:
- Shards optimizer states across DP ranks (like ZeRO-1/2)
- Manages FP32 master weights
- Handles gradient reduction and parameter gathering

### Step 5: Create Learning Rate Scheduler

```python
opt_param_scheduler = OptimizerParamScheduler(
    optimizer, max_lr, min_lr, lr_warmup_iters, lr_decay_iters, lr_decay_style, ...
)
```

Supports warmup + decay schedules (cosine, linear, inverse-sqrt, WSD).

### Step 6: Load Checkpoint

```python
if checkpoint_exists(args.load):
    load_checkpoint(model, optimizer, opt_param_scheduler, ...)
```

---

## 4.5 The Training Loop: `train()`

**File**: `megatron/training/training.py`, line 2499

The `train()` function is the main loop that runs for `train_iters` iterations:

```python
def train(forward_step_func, model, optimizer, opt_param_scheduler, 
          train_data_iterator, valid_data_iterator, ...):
    
    iteration = args.iteration  # Resume point from checkpoint
    
    # Configure forward-backward function for pipeline parallelism
    forward_backward_func = get_forward_backward_func()
    
    while iteration < args.train_iters:
        # === Core training step ===
        loss_dict, skipped_iter, grad_norm, ... = train_step(
            forward_step_func, train_data_iterator, model, optimizer, 
            opt_param_scheduler, config, forward_backward_func, iteration
        )
        
        # === Logging ===
        training_log(loss_dict, iteration, grad_norm, ...)
        
        # === Periodic evaluation ===
        if iteration % args.eval_interval == 0:
            evaluate_and_print_results(model, valid_data_iterator, ...)
        
        # === Periodic checkpointing ===
        if iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler, ...)
        
        iteration += 1
```

### What is `forward_backward_func`?

This is the pipeline parallelism schedule — it determines how micro-batches flow through the pipeline:

```python
from megatron.core.pipeline_parallel import get_forward_backward_func

# Returns one of:
# - forward_backward_no_pipelining()     # PP=1, simple forward then backward
# - forward_backward_pipelining_with_interleaving()  # Virtual PP (interleaved 1F1B)
# - forward_backward_pipelining_without_interleaving()  # Standard 1F1B
```

For PP=1 (no pipeline), it simply runs forward on all micro-batches, then backward on all.

---

## 4.6 The `train_step()` Function: One Iteration

**File**: `megatron/training/training.py`, line 1670

`train_step()` is the heart of training — one complete iteration:

```python
def train_step(forward_step_func, data_iterator, model, optimizer, 
               opt_param_scheduler, config, forward_backward_func, iteration):
    
    # 1. Zero gradients
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    
    # 2. Forward + Backward (across all micro-batches)
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=get_num_microbatches(),
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=False,
    )
    
    # 3. Optimizer step (all-reduce gradients, update weights)
    update_successful, grad_norm, num_zeros = optimizer.step()
    
    # 4. Update learning rate
    if update_successful:
        opt_param_scheduler.step(increment=num_tokens_processed)
    
    # 5. Average loss across micro-batches for logging
    loss_reduced = average_losses(losses_reduced)
    
    return loss_reduced, skipped_iter, grad_norm, ...
```

### Step-by-Step Breakdown

#### Step 1: Zero Gradients
```python
model_chunk.zero_grad_buffer()  # Custom gradient buffer (not .zero_grad())
optimizer.zero_grad()
```

Megatron uses custom gradient buffers (not PyTorch's default `.grad`) to enable:
- Gradient bucketing for efficient all-reduce
- Overlap of gradient reduction with backward pass
- Support for the distributed optimizer

#### Step 2: Forward + Backward

`forward_backward_func` handles the full forward-backward computation for all micro-batches. For no-pipeline case:

```
Micro-batch 1: forward → compute loss → backward
Micro-batch 2: forward → compute loss → backward
Micro-batch 3: forward → compute loss → backward
...
Micro-batch N: forward → compute loss → backward
```

Gradients accumulate across micro-batches. The actual `forward_step_func` is called once per micro-batch.

For pipeline parallelism, the schedule is more complex (1F1B, interleaved), but the interface is the same.

#### Step 3: Optimizer Step

```python
update_successful, grad_norm, num_zeros = optimizer.step()
```

This single line hides tremendous complexity:
1. **Finalize gradients**: All-reduce across DP ranks (or reduce-scatter for distributed optimizer)
2. **Gradient clipping**: Compute global grad norm, clip if needed
3. **Loss scaling**: For FP16, adjust the loss scale if overflow detected
4. **Weight update**: Adam/SGD/Muon step on FP32 master weights
5. **Copy back**: Copy updated FP32 weights back to FP16/BF16 model weights

If the gradient norm is NaN/Inf (overflow in FP16), `update_successful=False` and the step is skipped.

#### Step 4: Learning Rate Update

```python
opt_param_scheduler.step(increment=num_tokens_processed)
```

The scheduler tracks progress by **number of tokens consumed**, not iterations. This makes the schedule independent of micro-batch size or DP world size.

---

## 4.7 Micro-Batches vs. Global Batch

Understanding the batch hierarchy is critical:

```
Global Batch Size (GBS) = micro_batch_size × num_microbatches × data_parallel_size

Example: GBS = 4 × 8 × 64 = 2048 sequences
  - micro_batch_size = 4      (sequences per GPU per micro-batch)
  - num_microbatches = 8       (micro-batches per iteration per GPU)
  - data_parallel_size = 64    (number of DP replicas)
```

| Term | Scope | Size |
|------|-------|------|
| **Micro-batch** | One forward/backward on one GPU | `micro_batch_size` sequences |
| **Mini-batch** | All micro-batches on one GPU | `micro_batch_size × num_microbatches` |
| **Global batch** | All GPUs combined | `micro_batch_size × num_microbatches × dp_size` |

Gradients are accumulated across micro-batches locally, then all-reduced across DP ranks. The optimizer updates once per global batch.

### Batch Size Rampup

Megatron supports starting with a smaller batch size and linearly increasing it:

```bash
--rampup-batch-size 128 128 5000000  # Start at 128, increase by 128, reach full at 5M samples
```

This helps training stability in early iterations.

---

## 4.8 Loss Computation and Reporting

### Per-Micro-Batch Loss

The model returns per-token cross-entropy losses. The `loss_func` applies the loss mask and sums:

```python
loss = torch.sum(per_token_losses * loss_mask)  # Sum, not mean
num_tokens = loss_mask.sum()
```

### Averaging Across the Global Batch

After all micro-batches complete, `train_step` averages:

```python
# For each loss key (e.g., 'lm loss'):
# val = [(loss_1, ntokens_1), (loss_2, ntokens_2), ...]  (one per micro-batch)
val = torch.vstack(val).sum(dim=0)  # total_loss, total_tokens
torch.distributed.all_reduce(val, group=dp_group)  # Sum across DP ranks
loss_reduced[key] = val[0] / val[1]  # total_loss / total_tokens
```

This gives the correct average loss per token across the entire global batch, accounting for variable-length sequences and padding.

---

## 4.9 Pipeline Parallelism Schedules

When `pipeline_model_parallel_size > 1`, the `forward_backward_func` implements a pipeline schedule.

### No Pipeline (PP=1)

```
GPU: |--F1--|--F2--|--F3--|--F4--|--B4--|--B3--|--B2--|--B1--|
```

All forwards, then all backwards. Simple but doesn't overlap compute.

### 1F1B Schedule (Standard PP)

```
Stage 0: |--F1--|--F2--|--F3--|--F4--|--B1--|--F5--|--B2--|--F6--|--B3--|...
Stage 1:        |--F1--|--F2--|--F3--|--F4--|--B1--|--F5--|--B2--|...
Stage 2:               |--F1--|--F2--|--F3--|--F4--|--B1--|--F5--|...
Stage 3:                      |--F1--|--F2--|--F3--|--F4--|--B1--|...
```

After the warmup phase, each stage alternates one forward and one backward. This limits peak memory to `PP + 1` micro-batches (instead of `num_microbatches`).

### Virtual Pipeline Parallelism (Interleaved)

With `virtual_pipeline_model_parallel_size > 1`, each GPU holds multiple chunks of the model, further reducing pipeline bubbles.

---

## 4.10 The Training Log

After each step, `training_log()` reports metrics to TensorBoard/WandB:

- **Loss**: LM loss, MoE auxiliary losses, MTP losses
- **Learning rate**: Current LR from scheduler
- **Gradient norm**: Global L2 norm of gradients (before clipping)
- **Throughput**: Tokens/sec, samples/sec, TFLOP/s, MFU
- **Timing**: Forward time, backward time, optimizer time, data loading time
- **Memory**: GPU memory allocated/reserved

### FLOP Calculation

Megatron computes theoretical FLOPs using the standard formula:

```
FLOPs_per_iteration = 3 × (  # 3x for forward + backward + recomputation
    attention_flops +          # 4·b·s·h·(h + h·g/n + s/2) per layer
    mlp_flops +                # 4·e·b·s·h² per layer (e=expansion, typically 4)
    logit_flops                # 2·b·s·h·V
)
```

This is used to compute Model FLOPs Utilization (MFU):
```
MFU = achieved_TFLOPs / theoretical_peak_TFLOPs
```

---

## 4.11 Validation and Checkpointing

### Validation

Periodically (every `--eval-interval` iterations), the training loop runs evaluation:

```python
if iteration % args.eval_interval == 0:
    prefix = f'iteration {iteration}'
    evaluate_and_print_results(
        prefix, forward_step_func, valid_data_iterator, model, iteration, ...
    )
```

Evaluation runs the same `forward_step_func` but with `forward_only=True` (no backward pass).

### Checkpointing

Periodic saves (every `--save-interval` iterations):

```python
if iteration % args.save_interval == 0:
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 
                    num_floating_point_operations_so_far, ...)
```

Checkpoints include: model weights, optimizer state, LR scheduler state, RNG states, iteration number, and consumed samples count.

---

## 4.12 NaN/Inf Detection and Rerun

Megatron has a robust **rerun state machine** for handling training instabilities:

```python
# In loss_func:
rerun_state_machine.validate_result(
    result=loss,
    rejection_func=torch.isnan,    # Check for NaN
    message="found NaN in local forward loss",
)

# Spiky loss detection (loss > 10× max observed)
rerun_state_machine.validate_result(
    result=loss,
    rejection_func=partial(is_unexpectedly_large, threshold=10),
    message="Spiky loss",
)
```

When a problematic iteration is detected, the rerun machine can:
1. Skip the optimizer step
2. Rerun the iteration with different data
3. Log the event for debugging

---

## Key Files Covered

| File | Role |
|------|------|
| `pretrain_gpt.py` | Entry script: defines model_provider, forward_step, loss_func, datasets_provider |
| `model_provider.py` | GPTModel construction with specs |
| `megatron/training/training.py` | Training engine: pretrain(), train(), train_step(), training_log() |
| `megatron/training/initialize.py` | initialize_megatron(): distributed setup, args, seeds |
| `megatron/core/pipeline_parallel/` | get_forward_backward_func(): pipeline schedules |
| `megatron/core/num_microbatches_calculator.py` | Micro-batch count management and batch rampup |
| `megatron/training/checkpointing.py` | save_checkpoint(), load_checkpoint() |
| `megatron/core/transformer/module.py` | Float16Module: mixed-precision wrapper |
| `megatron/core/distributed/` | DDP, FSDP wrappers for data parallelism |
| `megatron/core/optimizer/` | Distributed optimizer, LR scheduler |
| `megatron/core/rerun_state_machine.py` | NaN/spiky loss detection and recovery |

---

## Interview-Ready Takeaways

1. **Separation of Concerns**: Megatron separates *what* to train (user callbacks: model_provider, forward_step, loss_func) from *how* to train (the training engine). This is why the same `train()` loop works for GPT, BERT, T5, multimodal, and even RL.

2. **Micro-batch Gradient Accumulation**: The global batch is split into micro-batches for memory efficiency. Gradients accumulate locally across micro-batches, then are all-reduced across DP ranks. One optimizer step per global batch.

3. **Loss is a Sum, Not a Mean**: Per-micro-batch loss is the *sum* of per-token losses (masked). The mean is computed by dividing by total tokens across all DP ranks. This handles variable-length sequences correctly.

4. **Pipeline Schedule**: `forward_backward_func` abstracts the pipeline schedule. For PP=1, it's simple sequential forward-then-backward. For PP>1, it's 1F1B (one forward, one backward) to limit memory to ~PP micro-batches.

5. **Custom Gradient Buffers**: Megatron doesn't use PyTorch's `.grad`. It maintains contiguous gradient buffers per DDP bucket for efficient all-reduce and supports overlapping gradient reduction with backward computation.

6. **Optimizer Step Complexity**: `optimizer.step()` does far more than weight update — it handles gradient all-reduce, gradient clipping, loss scale adjustment (FP16), and FP32→FP16/BF16 weight copy.

7. **Token-Based Scheduling**: The LR scheduler advances by tokens consumed, not iterations. This makes the training schedule invariant to changes in batch size (e.g., during batch rampup).

8. **Batch Size Rampup**: Starting with small batches and linearly increasing improves training stability. Megatron handles this transparently via `update_num_microbatches()`.

9. **NaN Recovery**: The rerun state machine detects NaN/Inf losses and spiky losses, automatically skipping bad iterations. This is critical for stable training at scale.

10. **MFU Calculation**: Model FLOPs Utilization = achieved_TFLOPs / peak_TFLOPs. Megatron computes theoretical FLOPs per iteration using known formulas for attention (4bsh(h + hg/n + s/2)) and MLP (4e·bsh²), multiplied by 3 for forward+backward+recomputation.

---

*Next Chapter: [Chapter 5 — Distributed Training & Parallelism](ch05_distributed_training.md) — tensor parallelism, pipeline parallelism, data parallelism (with math), context parallelism, and expert parallelism.*
