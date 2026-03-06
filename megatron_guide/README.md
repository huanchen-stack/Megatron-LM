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

### Part I — Data & Preprocessing
| # | Chapter | Description |
|---|---------|-------------|
| 0 | [Introduction & Codebase Overview](./ch00_introduction.md) | Project structure, two-component architecture (Megatron-LM vs Megatron Core), how to navigate the codebase |
| 1 | [Data Preprocessing Pipeline](./ch01_data_preprocessing.md) | Raw text → tokenized binary datasets. Tokenizers, `preprocess_data.py`, IndexedDataset format (.bin/.idx) |
| 2 | [Data Loading & Dataset Construction](./ch02_data_loading.md) | How binary datasets become training batches. GPTDataset three-index trick, blending, sampling |

### Part II — Model & Training
| # | Chapter | Description |
|---|---------|-------------|
| 3 | [Model Architecture Deep Dive](./ch03_model_architecture.md) | GPTModel internals: TransformerBlock, attention (GQA, MLA, RoPE), MLP (SwiGLU), MoE, the Spec pattern |
| 4 | [The Pretraining Loop](./ch04_pretraining_loop.md) | `pretrain()` → `train()` → `train_step()` → forward/backward. Micro-batch accumulation, pipeline schedules |
| 5 | [Distributed Training & Parallelism](./ch05_distributed_training.md) | 5D parallelism (TP/PP/DP/CP/EP). Deep dive on DP math: all-reduce, reduce-scatter, ZeRO stages |
| 6 | [Optimizers, Mixed Precision & Memory](./ch06_optimizers_mixed_precision.md) | Distributed optimizer, three-copy pattern, FP16/BF16/FP8, gradient clipping, Adam/Muon, LR scheduling, activation recomputation |
| 7 | [Checkpointing & Model Conversion](./ch07_checkpointing_conversion.md) | ShardedTensor, distributed save/load, checkpoint resharding magic, HuggingFace ↔ Megatron conversion, async saving |

### Part III — Post-Training (Interview Focus)
| # | Chapter | Description |
|---|---------|-------------|
| 8 | [Post-Training: SFT & LoRA](./ch08_post_training_sft.md) | SFT datasets (conversation packing, cu_seqlens, loss masking), chat templates, LoRA/PEFT/QLoRA deep dive |
| 9 | [Post-Training: RLHF & GRPO](./ch09_post_training_rlhf_grpo.md) | GRPO algorithm (math, advantages, clipped surrogate), agent/environment architecture, rollout collection, memory management |
| 10 | [Post-Training: Quantization, Pruning & Distillation](./ch10_quantization_pruning_distillation.md) | PTQ/QAT (FP8, NVFP4), Minitron pruning, knowledge distillation, EAGLE3 speculative decoding |
| 11 | [Model Export & Deployment](./ch11_export_deployment.md) | ModelOpt export pipeline, TRT-LLM/vLLM/SGLang targets, checkpoint format landscape, end-to-end examples |

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
│               POST-TRAINING: SFT & LoRA                    │
│  Conversation packing, loss masking, LoRA/PEFT             │
│  Ch 8: megatron/training/datasets/, megatron/post_training/│
├─────────────────────────────────────────────────────────────┤
│               POST-TRAINING: RLHF & GRPO                   │
│  GRPO algorithm, rollout collection, agent/environment     │
│  Ch 9: megatron/rl/, train_rl.py                           │
├─────────────────────────────────────────────────────────────┤
│            QUANTIZATION, PRUNING & DISTILLATION             │
│  PTQ/QAT, Minitron pruning, knowledge distillation         │
│  Ch 10: examples/post_training/modelopt/                   │
├─────────────────────────────────────────────────────────────┤
│                MODEL EXPORT & DEPLOYMENT                    │
│  Export → TRT-LLM / vLLM / SGLang → Serve                 │
│  Ch 11: megatron/core/export/, ModelOpt                    │
└─────────────────────────────────────────────────────────────┘
```

## Target Audience

- ML engineers preparing for technical interviews at AI companies
- Researchers looking to understand production training infrastructure
- Engineers transitioning from small-scale to large-scale training

**Assumed knowledge**: Basic deep learning (backprop, transformers, attention), Python/PyTorch.
**Not required**: Prior experience with distributed training, parallelism strategies, or the Megatron codebase.
