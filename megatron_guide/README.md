# Megatron-LM Training Guide: From Data to Deployment

A comprehensive, code-level walkthrough of large language model training using NVIDIA's Megatron-LM codebase. Written for engineers who understand basic deep learning concepts but are new to production-scale training infrastructure.

## How to Read This Guide

Each chapter follows the same structure:
1. **Conceptual overview** — What this stage does and why it matters
2. **Code entry points** — Where to start reading in the codebase
3. **Call chain walkthrough** — Step-by-step trace through the code with explanations
4. **Key files covered** — Summary table of all files discussed
5. **Interview-ready takeaways** — Concise points you should be able to articulate

## Chapters

| # | Chapter | Description |
|---|---------|-------------|
| 0 | [Introduction & Codebase Overview](./ch00_introduction.md) | Project structure, two-component architecture (Megatron-LM vs Megatron Core), how to navigate the codebase |
| 1 | [Data Preprocessing Pipeline](./ch01_data_preprocessing.md) | Raw text → tokenized binary datasets. Tokenizers, `preprocess_data.py`, IndexedDataset format |
| 2 | [Data Loading & Dataset Construction](./ch02_data_loading.md) | How binary datasets become training batches. GPTDataset, blending, sampling, SFT and FIM datasets |
| 3 | [Model Architecture Deep Dive](./ch03_model_architecture.md) | GPTModel internals: TransformerBlock, attention, MLP, MoE, and the TransformerSpec pattern |
| 4 | [The Pretraining Loop](./ch04_pretraining_loop.md) | What happens when you run `torchrun pretrain_gpt.py`. The full training step from data to gradient update |
| 5 | [Distributed Training & Parallelism](./ch05_distributed_training.md) | Tensor, pipeline, data, context, and expert parallelism. Process groups and communication patterns |
| 6 | [Optimizers & Mixed Precision](./ch06_optimizers_mixed_precision.md) | Distributed optimizer (ZeRO), gradient clipping, FP16/BF16/FP8/FP4, activation recomputation |
| 7 | [Checkpointing & Model Conversion](./ch07_checkpointing.md) | Saving/loading distributed checkpoints, HuggingFace ↔ Megatron conversion, checkpoint resharding |
| 8 | [Post-Training: SFT & RLHF](./ch08_post_training.md) | Supervised fine-tuning datasets, GRPO reinforcement learning, the RL training loop |
| 9 | [Model Optimization & Deployment](./ch09_optimization_deployment.md) | Quantization (FP8/FP4), pruning, knowledge distillation, EAGLE3 speculative decoding, model export |

## Quick Reference: The Training Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION                         │
│  Raw Text → Tokenize → IndexedDataset (.bin/.idx)          │
│  Ch 1: tools/preprocess_data.py                            │
├─────────────────────────────────────────────────────────────┤
│                    DATA LOADING                             │
│  .bin/.idx → GPTDataset → BlendedDataset → DataLoader      │
│  Ch 2: megatron/core/datasets/                             │
├─────────────────────────────────────────────────────────────┤
│                    MODEL BUILDING                           │
│  TransformerConfig → GPTModel (Attention + MLP layers)     │
│  Ch 3: megatron/core/models/gpt/                           │
├─────────────────────────────────────────────────────────────┤
│                    PRETRAINING                              │
│  pretrain() → train() → train_step() → forward/backward   │
│  Ch 4: megatron/training/training.py                       │
├─────────────────────────────────────────────────────────────┤
│                    DISTRIBUTED TRAINING                     │
│  TP / PP / DP / CP / EP — splitting work across GPUs       │
│  Ch 5: megatron/core/{tensor,pipeline}_parallel/           │
├─────────────────────────────────────────────────────────────┤
│                    OPTIMIZATION                             │
│  Distributed optimizer, mixed precision, grad clipping     │
│  Ch 6: megatron/core/optimizer/                            │
├─────────────────────────────────────────────────────────────┤
│                    CHECKPOINTING                            │
│  Save/load distributed state, convert between formats      │
│  Ch 7: megatron/core/dist_checkpointing/                   │
├─────────────────────────────────────────────────────────────┤
│                    POST-TRAINING                            │
│  SFT fine-tuning, GRPO/RLHF alignment                     │
│  Ch 8: megatron/rl/, megatron/training/datasets/           │
├─────────────────────────────────────────────────────────────┤
│                    DEPLOYMENT                               │
│  Quantize → Prune → Distill → Export → Serve              │
│  Ch 9: examples/post_training/modelopt/                    │
└─────────────────────────────────────────────────────────────┘
```
