# Chapter 0: Introduction & Codebase Overview

## What Is Megatron-LM?

Training a large language model is fundamentally different from training a ResNet on ImageNet. When your model has hundreds of billions of parameters, it won't fit on a single GPU. When your training dataset is terabytes of text, you can't just `json.load()` it into memory. When you need thousands of GPUs working in concert, vanilla PyTorch `DistributedDataParallel` won't cut it.

**Megatron-LM** is NVIDIA's answer to these challenges. It's an open-source PyTorch library that provides the specialized infrastructure needed to train transformer models at scales from 2B to 462B+ parameters across thousands of GPUs, achieving up to 47% Model FLOP Utilization (MFU) on H100 clusters.

The project lives at [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and consists of two distinct components:

| Component | Purpose | Who It's For |
|-----------|---------|-------------|
| **Megatron Core** (`megatron/core/`) | Composable library of GPU-optimized building blocks | Framework developers building custom training pipelines |
| **Megatron-LM** (everything else) | Reference training scripts using Megatron Core | Research teams, learning distributed training, quick experiments |

Think of Megatron Core as the engine and Megatron-LM as the reference car built with that engine. You can use the engine to build your own car, or drive the reference one as-is.

## Project Structure

Here's how the codebase is organized, with the purpose of each top-level directory:

```
Megatron-LM/
├── megatron/
│   ├── core/                    # THE LIBRARY — GPU-optimized building blocks
│   │   ├── models/              #   Model architectures (GPT, BERT, T5, Mamba, Multimodal)
│   │   ├── transformer/         #   Transformer layers (attention, MLP, MoE)
│   │   ├── tensor_parallel/     #   Tensor parallelism primitives
│   │   ├── pipeline_parallel/   #   Pipeline parallelism schedules
│   │   ├── distributed/         #   DDP, FSDP wrappers
│   │   ├── optimizer/           #   Distributed optimizer, gradient clipping
│   │   ├── datasets/            #   Dataset classes, data loading infrastructure
│   │   ├── dist_checkpointing/  #   Distributed checkpoint save/load
│   │   ├── inference/           #   Inference engine
│   │   └── export/              #   Model export (TensorRT-LLM)
│   ├── training/                # Training loop, arguments, initialization
│   │   ├── training.py          #   THE training loop (~3600 lines)
│   │   ├── arguments.py         #   All command-line arguments
│   │   ├── initialize.py        #   Distributed training initialization
│   │   ├── checkpointing.py     #   Checkpoint management
│   │   └── datasets/            #   SFT, FIM dataset wrappers
│   ├── post_training/           # Post-training (quantization, distillation via ModelOpt)
│   ├── rl/                      # Reinforcement learning (GRPO/RLHF)
│   ├── legacy/                  # Legacy code (being phased out)
│   └── inference/               # Legacy inference code
├── pretrain_gpt.py              # GPT pretraining entry point
├── pretrain_bert.py             # BERT pretraining entry point
├── pretrain_t5.py               # T5 pretraining entry point
├── pretrain_vlm.py              # Vision-language model pretraining
├── pretrain_mamba.py            # Mamba (SSM) pretraining
├── train_rl.py                  # RL post-training entry point
├── model_provider.py            # Model construction factory
├── gpt_builders.py              # GPT model builder configurations
├── examples/                    # Ready-to-use examples for various models
├── tools/                       # Data preprocessing, checkpoint conversion
└── tests/                       # Test suite
```

The key insight: almost everything interesting happens inside `megatron/core/`. The top-level `pretrain_*.py` scripts are thin wrappers that define three things and hand them to the training loop:
1. **How to build the model** (a `model_provider` function)
2. **How to build datasets** (a `train_valid_test_datasets_provider` function)
3. **How to do a forward step** (a `forward_step` function)

## The Training Lifecycle

Every LLM training pipeline follows the same lifecycle. Here's where each stage lives in Megatron:

```
Stage 1: DATA PREPARATION
  tools/preprocess_data.py → produces .bin/.idx binary files
  Covered in: Chapter 1

Stage 2: DATA LOADING
  megatron/core/datasets/ → loads .bin/.idx into training batches
  Covered in: Chapter 2

Stage 3: MODEL CONSTRUCTION
  megatron/core/models/gpt/ → builds the transformer model
  Covered in: Chapter 3

Stage 4: PRETRAINING
  pretrain_gpt.py → megatron/training/training.py → the training loop
  Covered in: Chapter 4

Stage 5: DISTRIBUTED TRAINING
  megatron/core/{tensor,pipeline}_parallel/ → splits work across GPUs
  Covered in: Chapter 5

Stage 6: OPTIMIZATION
  megatron/core/optimizer/ → distributed optimizer, mixed precision
  Covered in: Chapter 6

Stage 7: CHECKPOINTING
  megatron/core/dist_checkpointing/ → save/load/convert checkpoints
  Covered in: Chapter 7

Stage 8: POST-TRAINING (SFT & LoRA)
  megatron/training/datasets/sft_dataset.py → supervised fine-tuning
  Covered in: Chapter 8

Stage 9: POST-TRAINING (RLHF & GRPO)
  megatron/rl/ + train_rl.py → reinforcement learning from human feedback
  Covered in: Chapter 9

Stage 10: COMPRESSION (Quantization, Pruning, Distillation)
  examples/post_training/modelopt/ → quantize, prune, distill
  Covered in: Chapter 10

Stage 11: DEPLOYMENT
  megatron/core/export/ + ModelOpt → export to TRT-LLM/vLLM/SGLang
  Covered in: Chapter 11
```

## Your First Training Loop: The "Hello World"

The best way to understand Megatron is to start with the simplest possible training script. Open `examples/run_simple_mcore_train_loop.py` — it's about 270 lines and shows the entire training lifecycle without any of the production complexity.

### Step 1: Initialize Distributed Training

```python
def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1):
    parallel_state.destroy_model_parallel()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # This is the Megatron magic — creates TP, PP, DP process groups
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )
```

Two things happen here that go beyond vanilla PyTorch:
1. `torch.distributed.init_process_group()` — standard PyTorch distributed setup
2. `parallel_state.initialize_model_parallel()` — Megatron-specific: partitions GPUs into tensor-parallel, pipeline-parallel, and data-parallel groups. This is the foundation of everything in Chapters 5-6.

### Step 2: Build the Model

```python
def model_provider():
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
    )
    return gpt_model
```

Notice two key patterns:
- **TransformerConfig**: A single config object that controls everything — number of layers, hidden size, attention heads, parallelism settings, precision, etc. This is the central control plane for model architecture (Chapter 3).
- **TransformerLayerSpec**: Instead of hardcoding layer implementations, Megatron uses a "spec" pattern that describes which modules to use. This allows swapping implementations (e.g., local attention vs. flash attention) without changing model code.

### Step 3: Load Training Data

```python
def get_train_data_iterator():
    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=MegatronTokenizer.from_pretrained(
            metadata_path={"library": "null-text"}, vocab_size=_SEQUENCE_LENGTH,
        ),
    )
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()
    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)
    return iter(train_dataloader)
```

This uses `MockGPTDataset` (random data) for simplicity, but in production you'd use `GPTDataset` with real preprocessed data. The `BlendedMegatronDatasetBuilder` is the standard entry point for constructing datasets — it handles blending multiple data sources, splitting into train/valid/test, and caching (Chapter 2).

### Step 4: The Training Loop

```python
forward_backward_func = get_forward_backward_func()

for iteration in range(5):
    optim.zero_grad()

    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=train_iterator,
        model=gpt_model,
        num_microbatches=1,
        seq_length=_SEQUENCE_LENGTH,
        micro_batch_size=8,
        decoder_seq_length=_SEQUENCE_LENGTH,
        forward_only=False,
    )

    finalize_model_grads([gpt_model])
    optim.step()
```

This is the critical part. Instead of calling `model(data)` and `loss.backward()` directly, Megatron uses `forward_backward_func` — a function returned by `get_forward_backward_func()`. This function implements the pipeline-parallel schedule (1F1B, interleaved, etc.) and handles:
- Splitting batches into microbatches
- Scheduling forward and backward passes across pipeline stages
- Managing gradient accumulation

After the forward-backward pass, `finalize_model_grads()` synchronizes gradients across data-parallel and tensor-parallel groups.

### Step 5: Checkpointing

```python
sharded_state_dict = model.sharded_state_dict(prefix="")
dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=ckpt_path)
```

Each rank saves only its own shard of the model. The `sharded_state_dict()` method returns a description of what each rank owns, and `dist_checkpointing` coordinates the distributed save across all ranks (Chapter 7).

## Why Not Just Use PyTorch?

If you're wondering why all this complexity is necessary, here's what naive PyTorch `DistributedDataParallel` doesn't give you:

| Challenge | Naive PyTorch | Megatron |
|-----------|--------------|---------|
| Model doesn't fit on one GPU | Out of memory | Tensor parallelism splits layers across GPUs |
| Model doesn't fit on 8 GPUs | Still OOM | Pipeline parallelism splits layers across nodes |
| GPU utilization is low | Bubble time in pipeline | Interleaved schedules minimize bubbles |
| Gradient sync is slow | Blocking all-reduce | Overlapped communication with computation |
| FP16 training is unstable | Manual loss scaling | Built-in grad scaler with dynamic adjustment |
| Checkpoint is 1TB | Single process save | Distributed checkpointing, each rank saves its shard |
| Data loading is a bottleneck | DataLoader with JSON | Binary memory-mapped datasets with O(1) access |

Megatron's value is that it solves all of these simultaneously and they all compose correctly. Getting tensor parallelism + pipeline parallelism + data parallelism + mixed precision + distributed checkpointing to work together is the hard part, and that's what Megatron Core provides.

## Key Files Covered

| File | Purpose |
|------|---------|
| `README.md` | Project overview, structure, performance benchmarks |
| `examples/run_simple_mcore_train_loop.py` | Minimal training loop — the "Hello World" of Megatron |
| `megatron/core/parallel_state.py` | Process group initialization for all parallelism types |
| `megatron/core/transformer/transformer_config.py` | Central config for model architecture |
| `megatron/core/models/gpt/gpt_model.py` | GPT model class |
| `megatron/core/datasets/gpt_dataset.py` | Dataset for GPT pretraining |
| `megatron/core/pipeline_parallel/schedules.py` | Forward-backward scheduling |
| `megatron/core/dist_checkpointing/` | Distributed checkpoint save/load |

## Interview-Ready Takeaways

1. **Megatron-LM has two components**: Megatron Core (the composable library) and Megatron-LM (reference training scripts that use it). Most of the interesting code is in `megatron/core/`.

2. **The training lifecycle is**: data prep (tokenize to binary) → data loading (memory-mapped datasets) → model building (TransformerConfig + LayerSpec) → training (forward-backward schedules) → checkpointing (distributed sharded saves).

3. **The `model_provider` pattern**: Training scripts don't build models directly. They provide a factory function that the training loop calls. This separates model architecture from training infrastructure.

4. **Forward-backward is not `loss.backward()`**: Megatron uses specialized scheduling functions that implement pipeline parallelism, microbatch splitting, and gradient accumulation in one coordinated operation.

5. **`parallel_state` is the foundation**: All parallelism in Megatron starts with `parallel_state.initialize_model_parallel()`, which partitions GPUs into tensor-parallel, pipeline-parallel, and data-parallel groups. Every distributed operation references these groups.

6. **Binary datasets are essential at scale**: You can't read JSON at training time with 1000 GPUs. Megatron preprocesses data into memory-mapped binary files (`.bin/.idx`) that support O(1) random access with zero deserialization.

7. **Everything composes**: The hard part isn't any single technique — it's making TP + PP + DP + mixed precision + distributed checkpointing all work together correctly. That's Megatron Core's core value.

---

*Next up: [Chapter 1 — Data Preprocessing Pipeline](./ch01_data_preprocessing.md) dives into how raw text becomes the binary datasets that Megatron trains on.*
