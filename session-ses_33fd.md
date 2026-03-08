# ML interview prep: Megatron codebase

**Session ID:** ses_33fdd34f9ffey2kHpbbGH03fgQ
**Created:** 3/5/2026, 2:33:51 PM
**Updated:** 3/5/2026, 2:35:52 PM

---

## User

[analyze-mode]
ANALYSIS MODE. Gather context before diving deep:

CONTEXT GATHERING (parallel):
- 1-2 explore agents (codebase patterns, implementations)
- 1-2 librarian agents (if external library involved)
- Direct tools: Grep, AST-grep, LSP for targeted searches

IF COMPLEX - DO NOT STRUGGLE ALONE. Consult specialists:
- **Oracle**: Conventional problems (architecture, debugging, complex logic)
- **Artistry**: Non-conventional problems (different approach needed)

SYNTHESIZE findings before proceeding.

---

i want to prepare for an interview and here's the guideline Deep dive: Technical Interview - Machine Learning
Succeeding in this role requires some specific technical capabilities, and we want to see
your skills in action. That’s where these two technical interviews come in:
●
●
Coding technical interview
Concept-focused technical interview
These interviews require the CodeSignal application.
What we’re looking for
For the coding technical interview, be prepared to:
●
●
●
●
Work through an algorithmic coding problem in Python, Java, Scala, C++ or Go
using CodeSignal
Answer questions about your approach to model implementation in business
settings
Receive and adapt to feedback—the ability to recognize and fix mistakes is
important
Discuss the reasoning behind your solutions
For the concept-focused technical interview, be prepared to:
●
Answer questions on diﬀerent areas of machine learning, such as the model
development cycle, deployment areas and feature engineering
●
Discuss and answer questions about your experience with machine learning
How to prepare
In addition to reviewing the topics above, you may also want to refresh yourself on the
CodeSignal platform, which you can access at https://codesignal.com/.
When your interview kicks oﬀ, your interviewer will provide instructions on how to sign in to
and use CodeSignal. If you have any questions in the meantime, reach out to your recruiter
for help.
#ᴅʟᴘ
_
MICRODOT [{'title': 'Data Security Classification'
,
'text': 'Confidential'}, {'title': 'doc
id'
_
,
'text': '1fTkR8oNQjadgbI
_
AqR9KeF49HsPFZNF1sS13qGN-St4'}]_
END
Deep dive: Technical Interview - Applied Research
During this interview, we’ll ask you a series of knowledge-based questions about applied
research and the technical skills that you’ll need in the role.
What we’re looking for
This conversation is the time to showcase your background and skills in applied research.
We’ll also use this interview to gauge your command of concepts in areas such as:
●
●
●
●
●
Large language models (LLMs) and natural language processing (NLP)
Behavioral models
Optimization
Fine tuning
Data preparation
You should expect both general and role-specific questions on linear algebra, distributed
systems and deep learning theory.
How to prepare
Come prepared to discuss your research expertise and how your experience will help you in
this role.  do you think this megatron codebase is a good place to kind of start  and learn about training (from data to pretrain to post train)? I have zero knowledge on training except for the 5d parallelism

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 15.6s)

_Thinking:_

The user is preparing for a technical interview focused on ML/applied research, and they're asking whether the Megatron-LM codebase is a good place to learn about the full training pipeline (data preparation → pretraining → post-training). They have some knowledge of 5D parallelism but want to understand the broader picture.

This is a research/evaluation intent - they want my assessment of whether Megatron-LM is a good learning resource for their interview prep. Let me explore the codebase to understand what it covers, then give them an honest assessment.

Let me fire some explore agents to understand the codebase structure and what it covers in terms of:


- Data preparation pipeline
- Pretraining infrastructure
- Post-training (fine-tuning, RLHF, etc.)
- Model architecture implementations

This is an evaluation/research question, not an implementation request. I should explore the codebase and give an honest assessment.

I detect **evaluation** intent — you want my honest assessment of whether Megatron-LM covers the full training lifecycle for your interview prep. Let me explore the codebase structure to give you a grounded answer.

Firing parallel exploration to map what Megatron-LM actually covers:

**Tool: task**

**Input:**
```json
{
  "subagent_type": "explore",
  "run_in_background": true,
  "load_skills": [],
  "description": "Map Megatron-LM data preparation pipeline",
  "prompt": "I need to understand what Megatron-LM offers for DATA PREPARATION and DATA PROCESSING. The user wants to learn the full training pipeline (data → pretrain → post-train).\n\nFind:\n1. Data preprocessing scripts/tools (tokenization, data formatting, dataset creation)\n2. Any data pipeline infrastructure (dataloaders, data sampling, data mixing)\n3. Dataset classes and how training data is fed to models\n4. Any documentation about data preparation workflows\n\nSearch in the root directory, megatron/ directory, tools/, scripts/, examples/, docs/. Look for files related to: tokenizer, dataset, dataloader, data_prep, preprocessing, blendable, indexed_dataset.\n\nReturn: file paths with brief descriptions of what each covers. Focus on coverage completeness — does it cover the full data pipeline?"
}
```

**Output:**
```
Background task launched.

Task ID: bg_c5533415
Description: Map Megatron-LM data preparation pipeline
Agent: explore
Status: running

System notifies on completion. Use `background_output` with task_id="bg_c5533415" to check.

<task_metadata>
session_id: ses_33fdcf893ffeEiiqw2ylzif0y7
</task_metadata>

to continue: task(session_id="ses_33fdcf893ffeEiiqw2ylzif0y7", prompt="...")
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 7.9s)

**Tool: task**

**Input:**
```json
{
  "subagent_type": "explore",
  "run_in_background": true,
  "load_skills": [],
  "description": "Map Megatron-LM training and post-training capabilities",
  "prompt": "I need to understand what Megatron-LM covers for PRETRAINING and POST-TRAINING (fine-tuning, RLHF, alignment, instruction tuning, etc.).\n\nFind:\n1. Pretraining entry points and training loops (pretrain scripts, training.py, etc.)\n2. Model architectures supported (GPT, BERT, T5, LLaMA, vision models, multimodal)\n3. Post-training: any fine-tuning, SFT, RLHF, DPO, alignment, instruction tuning code\n4. Optimizer implementations, learning rate schedulers\n5. Loss functions, evaluation, checkpointing\n6. Distributed training infrastructure beyond parallelism (mixed precision, gradient accumulation, etc.)\n7. Any examples/ directory showing end-to-end workflows\n\nSearch broadly: megatron/, examples/, pretrain_*.py, finetune_*.py, tools/, scripts/.\n\nReturn: file paths organized by category (pretrain vs post-train vs infra). Explicitly note what IS and IS NOT covered."
}
```

**Output:**
```
Background task launched.

Task ID: bg_1d4aac8b
Description: Map Megatron-LM training and post-training capabilities
Agent: explore
Status: running

System notifies on completion. Use `background_output` with task_id="bg_1d4aac8b" to check.

<task_metadata>
session_id: ses_33fdcd9a7ffeHfOcT11bP6PRKV
</task_metadata>

to continue: task(session_id="ses_33fdcd9a7ffeHfOcT11bP6PRKV", prompt="...")
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 4.6s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM</path>
<type>directory</type>
<entries>
.coderabbit.yaml
.flake8
.git/
.github/
.gitignore
.gitlab-ci.yml
.gitlab/
.pre-commit-config.yaml
.pylintrc
.python-version
codecov.yml
CONTRIBUTING.md
docker/
docs/
examples/
gpt_builders.py
greptile.json
images/
LICENSE
mamba_builders.py
MANIFEST.in
megatron/
model_provider.py
pretrain_bert.py
pretrain_gpt.py
pretrain_mamba.py
pretrain_t5.py
pretrain_vlm.py
pyproject.toml
README.md
scripts/
setup.py
tasks/
tests/
tools/
train_rl.py
uv.lock

(37 entries)
</entries>
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/README*",
  "path": "/Users/huanchen/Desktop/Megatron-LM"
}
```

**Output:**
```
Found 33 file(s)

/Users/huanchen/Desktop/Megatron-LM/tests/functional_tests/test_cases/gpt/gpt_static_inference_tp1_pp1_16b_multiprompt_tokensmatch/README.md
/Users/huanchen/Desktop/Megatron-LM/tests/functional_tests/test_cases/gpt/gpt3_mcore_reruns_resume_check_grads/README.md
/Users/huanchen/Desktop/Megatron-LM/tests/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/rl/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/core/transformer/moe/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/core/optimizer/cpu_offloading/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/core/models/mimo/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/core/distributed/fsdp/src/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/core/distributed/README.md
/Users/huanchen/Desktop/Megatron-LM/megatron/core/README_STRAGGLER.md
/Users/huanchen/Desktop/Megatron-LM/megatron/core/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/t5/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/rl/environments/countdown/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/rl/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/post_training/modelopt/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/multimodal/nvlm/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/multimodal/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/mixtral/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/llama/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/mamba/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/inference/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/gpt3/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/gptoss/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/export/trtllm_export/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/export/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/bert/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/academic_paper_scripts/sc21/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/academic_paper_scripts/msdp/README.md
/Users/huanchen/Desktop/Megatron-LM/examples/academic_paper_scripts/detxoify_lm/README.md
/Users/huanchen/Desktop/Megatron-LM/docs/discussions/README.md
/Users/huanchen/Desktop/Megatron-LM/README.md
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/pretrain_*.py",
  "path": "/Users/huanchen/Desktop/Megatron-LM"
}
```

**Output:**
```
Found 5 file(s)

/Users/huanchen/Desktop/Megatron-LM/pretrain_t5.py
/Users/huanchen/Desktop/Megatron-LM/pretrain_gpt.py
/Users/huanchen/Desktop/Megatron-LM/pretrain_vlm.py
/Users/huanchen/Desktop/Megatron-LM/pretrain_bert.py
/Users/huanchen/Desktop/Megatron-LM/pretrain_mamba.py
```

**Tool: glob**

**Input:**
```json
{
  "pattern": "**/finetune*.py",
  "path": "/Users/huanchen/Desktop/Megatron-LM"
}
```

**Output:**
```
Found 3 file(s)

/Users/huanchen/Desktop/Megatron-LM/tasks/finetune_utils.py
/Users/huanchen/Desktop/Megatron-LM/examples/post_training/modelopt/finetune.py
/Users/huanchen/Desktop/Megatron-LM/examples/academic_paper_scripts/detxoify_lm/finetune_gpt.py
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 6.0s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron</path>
<type>directory</type>
<entries>
core/
inference/
legacy/
post_training/
rl/
training/

(6 entries)
</entries>

[Project README: /Users/huanchen/Desktop/Megatron-LM/README.md]
<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.15.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts. Best for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision support (FP16, BF16, FP8, FP4), and model architectures. Best for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.


## Quick Start

Install Megatron Core with pip:

1. Install Megatron Core with required dependencies:

    ```bash
    pip install --no-build-isolation megatron-core[mlm,dev]
    ```

2. Clone repository for examples:

    ```bash
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    pip install --no-build-isolation .[mlm,dev]
    ```


# Latest News

- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.
- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

<details>
<summary>Previous News</summary>

- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.

</details>


# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```


# Performance Benchmarking

For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
- **Production ready**: Full training pipeline with checkpointing and fault tolerance
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)


# Roadmaps

- **[MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements


# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, we appreciate that you use the following citations:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```

```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/examples"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/examples</path>
<type>directory</type>
<entries>
__init__.py
academic_paper_scripts/
bert/
export/
gpt3/
gptoss/
inference/
llama/
mamba/
mimo/
mixtral/
multimodal/
post_training/
rl/
run_simple_mcore_train_loop.py
t5/

(16 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/rl/README.md"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/rl/README.md</path>
<type>file</type>
<content>
1#NT|# Megatron-RL
2#KM|
3#XQ|## Status
4#ZM|08/27/2025: Megatron-RL is actively under development. While it is functional internally at NVIDIA, it is not yet usable by external users because not all required code has been released. The available code and examples may change as development progresses. For a current roadmap of planned Megatron-RL features please see [#1776](https://github.com/NVIDIA/Megatron-LM/issues/1776).
5#BT|
6#MS|## Overview
7#RJ|Megatron-RL is adding native reinforcement learning (RL) based post-training to Megatron-LM. It provides a flexible library for defining RL environments and agents, extending the Megatron-LM training loop with RL algorithm support.
8#XW|
9#HQ|The bulk of the new library code is located in `megatron/rl`. However:
10#XQ|- Significant modifications have been made to the Megatron Core inference code.
11#SB|- Minor changes were made to the Megatron-LM training loop to enable Megatron-RL.
12#TX|
13#JK|Example environments for Megatron-RL can be found in `examples/rl`.
14#BY|
15#NH|Megatron-RL is designed for research teams exploring RL post-training of LLMs at scale on state-of-the-art foundation models with cutting-edge performance on the latest NVIDIA hardware.
16#VP|
17#KB|It is **not** intended as an enterprise framework and won't necessarily provide out-of-the-box support for any given open model. For those capabilities please refer to [Nemo RL](https://github.com/NVIDIA-NeMo/RL).
18#KS|
19#SY|## Design
20#YQ|
21#YB|The design philosophy of Megatron RL is to keep the agent/environment design as decoupled as possible from the underlying RL implementation.
22#BT|- The environment design defines the "Agent Environment" which takes as input a handle to an inference interface (i.e. something supporting `.generate(prompt, **generation_args)`) and must return experience rollouts along with rewards.
23#XM|- The RL training system handles batching inference requests, hosting inference, training and other orchestration tasks.
24#KW|
25#BP|Below we describe the different conceptual components and how they divide responsibility.
26#HK|
27#TH|### Agent and Environment (referred to as an Agent)
28#TQ|- Takes a handle to an `InferenceInterface`.
29#HN|- Returns `Rollout` or `EvaluationResponse` objects.
30#XP|- Responsible for sampling parameters, custom generation arguments (e.g., stop conditions, inline evaluation), etc.
31#SZ|
32#BQ|### Trainer/Evaluator
33#RY|- Manages the control flow for rollout generation and evaluation.
34#BN|- Coordinates with (or creates) the `InferenceInterface` and `Agent`s.
35#TX|
36#XB|### Inference Interface
37#RJ|- Provides the endpoint the `AgenticEnvironment` uses to run `.generate(prompt, **generation_args)`.
38#TM|- Can take many forms (e.g. Megatron, OpenAI, HF) and supports many configuration options.
39#MS|
40#TN|## Examples
41#JJ|See `examples/rl` for demonstrations of:
42#TP|- Custom `InferenceInterface` endpoints
43#QR|- Example `Agents`

(End of file - total 43 lines)
</content>

[Project README: /Users/huanchen/Desktop/Megatron-LM/megatron/rl/README.md]
# Megatron-RL

## Status
08/27/2025: Megatron-RL is actively under development. While it is functional internally at NVIDIA, it is not yet usable by external users because not all required code has been released. The available code and examples may change as development progresses. For a current roadmap of planned Megatron-RL features please see [#1776](https://github.com/NVIDIA/Megatron-LM/issues/1776).

## Overview
Megatron-RL is adding native reinforcement learning (RL) based post-training to Megatron-LM. It provides a flexible library for defining RL environments and agents, extending the Megatron-LM training loop with RL algorithm support.

The bulk of the new library code is located in `megatron/rl`. However:
- Significant modifications have been made to the Megatron Core inference code.
- Minor changes were made to the Megatron-LM training loop to enable Megatron-RL.

Example environments for Megatron-RL can be found in `examples/rl`.

Megatron-RL is designed for research teams exploring RL post-training of LLMs at scale on state-of-the-art foundation models with cutting-edge performance on the latest NVIDIA hardware.

It is **not** intended as an enterprise framework and won't necessarily provide out-of-the-box support for any given open model. For those capabilities please refer to [Nemo RL](https://github.com/NVIDIA-NeMo/RL).

## Design

The design philosophy of Megatron RL is to keep the agent/environment design as decoupled as possible from the underlying RL implementation.
- The environment design defines the "Agent Environment" which takes as input a handle to an inference interface (i.e. something supporting `.generate(prompt, **generation_args)`) and must return experience rollouts along with rewards.
- The RL training system handles batching inference requests, hosting inference, training and other orchestration tasks.

Below we describe the different conceptual components and how they divide responsibility.

### Agent and Environment (referred to as an Agent)
- Takes a handle to an `InferenceInterface`.
- Returns `Rollout` or `EvaluationResponse` objects.
- Responsible for sampling parameters, custom generation arguments (e.g., stop conditions, inline evaluation), etc.

### Trainer/Evaluator
- Manages the control flow for rollout generation and evaluation.
- Coordinates with (or creates) the `InferenceInterface` and `Agent`s.

### Inference Interface
- Provides the endpoint the `AgenticEnvironment` uses to run `.generate(prompt, **generation_args)`.
- Can take many forms (e.g. Megatron, OpenAI, HF) and supports many configuration options.

## Examples
See `examples/rl` for demonstrations of:
- Custom `InferenceInterface` endpoints
- Example `Agents`
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/examples/rl/README.md"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/examples/rl/README.md</path>
<type>file</type>
<content>
1#HZ|# Reinforcement Learning in megatron
2#KM|
3#JT|This is an example of GRPO implementation within megatron-lm.
4#PK|For implementation details check out `train_rl.py` and `megatron/rl/rl_utils.py`.
5#QT|For the environment details, check the `megatron.rl` module.
6#SY|
7#BZ|The following experiment will train the Qwen 2.5 32B model on the [DAPO17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset and will run evaluation on [AIME2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024).
8#NH|After 300 steps, you should get about 0.7 pass@32 on AIME with the average training reward of 0.6. 
9#JT|
10#NB|## Setup
11#TJ|
12#VS|You should be able to run qwen2p5_32b_grpo.sh using the `nvcr.io/nvidia/pytorch:25.06-py3` container with these additional dependencies:
13#BQ|
14#BV|```bash
15#SP|pip install flask-restful uvloop datasets evaluate
16#XN|```
17#HX|
18#SQ|Specify these environment variables and create the required directories:
19#YT|
20#BV|```bash
21#ZV|export CUDA_DEVICE_MAX_CONNECTIONS=1
22#ZP|
23#PZ|CHECKPOINT="" # <Specify path to the base model checkpoint>
24#VZ|RUN_DIR="" # <Specify path for bookkeeping>
25#WB|WANDB_PROJECT="" # <Specify>
26#NJ|WANDB_EXP_NAME="" # <Specify>
27#JJ|
28#HT|LOG_DIR=$RUN_DIR/logs
29#MB|DATA_CACHE_DIR=$RUN_DIR/data_cache
30#PB|CHECKPOINT_DIR=$RUN_DIR/checkpoints
31#MN|TB_DIR=$RUN_DIR/tensorboard
32#VP|```
33#QY|
34#QZ|## Convert the checkpoint
35#TX|
36#JX|You can convert a [Huggingface Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-32B) to megatron-lm format using the `megatron-lm/tools/checkpoint/convert.py` script:
37#RB|
38#BV|```bash
39#WH|TP=8
40#HJ|HF_FORMAT_DIR=<PATH_TO_HF_FORMAT_DIR>
41#TK|MEGATRON_FORMAT_DIR=<PATH_TO_MEGATRON_FORMAT_DIR>
42#WV|TOKENIZER_MODEL=HF_FORMAT_DIR
43#QB|
44#RX|python ./tools/checkpoint/convert.py \
45#VV|    --bf16 \
46#MN|    --model-type GPT \
47#RQ|    --loader llama_mistral \
48#JH|    --saver core \
49#BQ|    --target-tensor-parallel-size ${TP} \
50#RY|    --checkpoint-type hf \
51#RV|    --load-dir ${HF_FORMAT_DIR} \
52#RR|    --save-dir ${MEGATRON_FORMAT_DIR} \
53#TR|    --tokenizer-model ${TOKENIZER_MODEL} \
54#SJ|    --model-size qwen2.5 \
55#MH|    --loader-transformer-impl transformer_engine \
56#JQ|    --make-vocab-size-divisible-by 128 \
57#RJ|```
58#KR|
59#BT|## Experiment command
60#HQ|
61#SZ|NOTE: Depending on the environment you are running it the provided script might require minor changes.
62#RJ|
63#BV|```bash
64#XZ|
65#WN|COMMON_OPTIONS="\
66#ZR|    --tensor-model-parallel-size $TP  \
67#JH|    --pipeline-model-parallel-size $PP  \
68#TJ|    --use-mcore-models \
69#TB|    --transformer-impl transformer_engine \
70#VV|    --bf16 \
71#SW|    --te-rng-tracker \
72#JP|    --cuda-graph-impl local \
73#VR|    --inference-dynamic-batching-num-cuda-graphs 1 \
74#XP|    --inference-dynamic-batching-buffer-size-gb 20 \
75#YN|    --data-parallel-random-init \
76#XK|    --attention-backend flash \
77#PB|    --timing-log-level 1 \
78#VR|    --log-timers-to-tensorboard \
79#HX|    --initialize-socket-comms \
80#PS|    "
81#JQ|
82#YT|GRPO_CLAMP_EPS_LOWER=0.2
83#JT|GRPO_CLAMP_EPS_UPPER=0.28
84#KJ|MAX_INFERENCE_BS=32
85#ZR|GRPO_GROUP_SIZE=16
86#BQ|GRPO_PROMPTS_PER_STEP=64
87#WQ|GRPO_ITERATIONS=1
88#VN|GRPO_KL_BETA="0.0"
89#WW|TRAINING_BATCH_SIZE=1024
90#HK|MICRO_BATCH_SIZE=1
91#TK|MAX_SEQ_LENGTH=11999
92#JZ|
93#NJ|MODEL_OPTIONS="\
94#TN|  --ckpt-format torch \
95#SH|  --seq-length $MAX_SEQ_LENGTH \
96#TJ|  --inference-max-seq-length $MAX_SEQ_LENGTH \
97#XT|  --inference-max-requests $MAX_INFERENCE_BS \
98#VZ|  --pretrained-checkpoint $CHECKPOINT \
99#BR|  --untie-embeddings-and-output-weights \
100#YB|  --disable-bias-linear \
101#SW|  --add-qkv-bias \
102#VZ|  --normalization RMSNorm \
103#PJ|  --norm-epsilon 1e-5 \
104#SB|  --group-query-attention \
105#NH|  --num-query-groups 8 \
106#ZV|  --no-masked-softmax-fusion \
107#XQ|  --attention-softmax-in-fp32 \
108#PQ|  --attention-dropout 0.0 \
109#VB|  --hidden-dropout 0.0 \
110#VK|  --weight-decay 0.0 \
111#MT|  --position-embedding-type rope \
112#QN|  --rotary-percent 1.0 \
113#SS|  --rotary-base 1000000 \
114#XX|  --use-rotary-position-embeddings \
115#KR|  --swiglu \
116#KS|  --num-layers 64  \
117#JY|  --hidden-size 5120  \
118#TP|  --ffn-hidden-size 27648 \
119#YV|  --num-attention-heads 40  \
120#SK|  --max-position-embeddings 131072 \
121#MN|  --tokenizer-type HuggingFaceTokenizer \
122#RW|  --tokenizer-model unsloth/Qwen2.5-32B \
123#JV|  --lr 1e-6 \
124#SR|  --lr-warmup-samples 0 \
125#JQ|  --make-vocab-size-divisible-by 128 \
126#MV|  --clip-grad 1.0 \
127#NK|  --recompute-granularity selective \
128#HM|  --recompute-activations "
129#HP|
130#QS|ENV_DEPENDENT="\
131#KR|  --langrl-env-config "examples/rl/environment_configs/dapo.yaml" \
132#ZB|  --micro-batch-size $MICRO_BATCH_SIZE \
133#JH|  --global-batch-size $TRAINING_BATCH_SIZE \
134#XT|  --grpo-group-size $GRPO_GROUP_SIZE \
135#PK|  --grpo-prompts-per-step $GRPO_PROMPTS_PER_STEP \
136#YJ|  --grpo-iterations $GRPO_ITERATIONS \
137#XR|  --grpo-clamp-eps-lower $GRPO_CLAMP_EPS_LOWER \
138#RY|  --grpo-clamp-eps-upper $GRPO_CLAMP_EPS_UPPER \
139#BJ|  --grpo-kl-beta $GRPO_KL_BETA \
140#BB|  --env-config $ENV_CONFIG "
141#WX|
142#HQ|
143#JS|torchrun \
144#VW|    --nproc-per-node=8 \
145#VY|    --nnodes=8 \
146#YH|    train_rl.py \
147#QW|    --mock-data \
148#NR|    --distributed-timeout-minutes 60 \
149#NR|    --train-samples 48828125 \
150#YN|    --log-interval 10 \
151#MN|    --log-progress  \
152#HR|    --timing-log-option minmax \
153#RJ|    --log-params-norm \
154#SV|    --log-num-zeros-in-grad \
155#KJ|    --log-throughput \
156#PT|    --adam-beta1 0.9 \
157#VJ|    --adam-beta2 0.95 \
158#HB|    --adam-eps 1e-8 \
159#JH|    --no-create-attention-mask-in-dataloader \
160#RH|    --accumulate-allreduce-grads-in-fp32 \
161#WQ|    --calculate-per-token-loss \
162#TH|    --log-straggler \
163#YS|    --disable-straggler-on-startup \
164#XX|    --perform-rl-step \
165#RM|    --use-distributed-optimizer \
166#JW|    --straggler-minmax-count 16 \
167#KK|    --eval-interval 20 \
168#BX|    --rl-prompts-per-eval 32 \
169#YM|    --tensorboard-log-interval 1 \
170#VH|    --empty-unused-memory-level 2 \
171#QM|    --data-cache-path ${DATA_CACHE_DIR} \
172#WB|    --save $CHECKPOINT_DIR \
173#NZ|    --load $CHECKPOINT_DIR \
174#YX|    --tensorboard-dir $TB_DIR \
175#TJ|    --seed $SEED \
176#PS|    --sequence-parallel \
177#BY|    --finetune \
178#WB|    --save-interval 20 \
179#VQ|    --wandb-project $WANDB_PROJECT \
180#MS|    --wandb-exp-name $WANDB_EXP_NAME \
181#JX|    ${MODEL_OPTIONS} \
182#BB|    ${COMMON_OPTIONS} \
183#NY|    ${ENV_DEPENDENT} $@
184#PR|```

(End of file - total 184 lines)
</content>

[Project README: /Users/huanchen/Desktop/Megatron-LM/examples/rl/README.md]
# Reinforcement Learning in megatron

This is an example of GRPO implementation within megatron-lm.
For implementation details check out `train_rl.py` and `megatron/rl/rl_utils.py`.
For the environment details, check the `megatron.rl` module.

The following experiment will train the Qwen 2.5 32B model on the [DAPO17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) dataset and will run evaluation on [AIME2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024).
After 300 steps, you should get about 0.7 pass@32 on AIME with the average training reward of 0.6. 

## Setup

You should be able to run qwen2p5_32b_grpo.sh using the `nvcr.io/nvidia/pytorch:25.06-py3` container with these additional dependencies:

```bash
pip install flask-restful uvloop datasets evaluate
```

Specify these environment variables and create the required directories:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT="" # <Specify path to the base model checkpoint>
RUN_DIR="" # <Specify path for bookkeeping>
WANDB_PROJECT="" # <Specify>
WANDB_EXP_NAME="" # <Specify>

LOG_DIR=$RUN_DIR/logs
DATA_CACHE_DIR=$RUN_DIR/data_cache
CHECKPOINT_DIR=$RUN_DIR/checkpoints
TB_DIR=$RUN_DIR/tensorboard
```

## Convert the checkpoint

You can convert a [Huggingface Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-32B) to megatron-lm format using the `megatron-lm/tools/checkpoint/convert.py` script:

```bash
TP=8
HF_FORMAT_DIR=<PATH_TO_HF_FORMAT_DIR>
MEGATRON_FORMAT_DIR=<PATH_TO_MEGATRON_FORMAT_DIR>
TOKENIZER_MODEL=HF_FORMAT_DIR

python ./tools/checkpoint/convert.py \
    --bf16 \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --target-tensor-parallel-size ${TP} \
    --checkpoint-type hf \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --model-size qwen2.5 \
    --loader-transformer-impl transformer_engine \
    --make-vocab-size-divisible-by 128 \
```

## Experiment command

NOTE: Depending on the environment you are running it the provided script might require minor changes.

```bash

COMMON_OPTIONS="\
    --tensor-model-parallel-size $TP  \
    --pipeline-model-parallel-size $PP  \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --bf16 \
    --te-rng-tracker \
    --cuda-graph-impl local \
    --inference-dynamic-batching-num-cuda-graphs 1 \
    --inference-dynamic-batching-buffer-size-gb 20 \
    --data-parallel-random-init \
    --attention-backend flash \
    --timing-log-level 1 \
    --log-timers-to-tensorboard \
    --initialize-socket-comms \
    "

GRPO_CLAMP_EPS_LOWER=0.2
GRPO_CLAMP_EPS_UPPER=0.28
MAX_INFERENCE_BS=32
GRPO_GROUP_SIZE=16
GRPO_PROMPTS_PER_STEP=64
GRPO_ITERATIONS=1
GRPO_KL_BETA="0.0"
TRAINING_BATCH_SIZE=1024
MICRO_BATCH_SIZE=1
MAX_SEQ_LENGTH=11999

MODEL_OPTIONS="\
  --ckpt-format torch \
  --seq-length $MAX_SEQ_LENGTH \
  --inference-max-seq-length $MAX_SEQ_LENGTH \
  --inference-max-requests $MAX_INFERENCE_BS \
  --pretrained-checkpoint $CHECKPOINT \
  --untie-embeddings-and-output-weights \
  --disable-bias-linear \
  --add-qkv-bias \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --group-query-attention \
  --num-query-groups 8 \
  --no-masked-softmax-fusion \
  --attention-softmax-in-fp32 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --weight-decay 0.0 \
  --position-embedding-type rope \
  --rotary-percent 1.0 \
  --rotary-base 1000000 \
  --use-rotary-position-embeddings \
  --swiglu \
  --num-layers 64  \
  --hidden-size 5120  \
  --ffn-hidden-size 27648 \
  --num-attention-heads 40  \
  --max-position-embeddings 131072 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model unsloth/Qwen2.5-32B \
  --lr 1e-6 \
  --lr-warmup-samples 0 \
  --make-vocab-size-divisible-by 128 \
  --clip-grad 1.0 \
  --recompute-granularity selective \
  --recompute-activations "

ENV_DEPENDENT="\
  --langrl-env-config "examples/rl/environment_configs/dapo.yaml" \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $TRAINING_BATCH_SIZE \
  --grpo-group-size $GRPO_GROUP_SIZE \
  --grpo-prompts-per-step $GRPO_PROMPTS_PER_STEP \
  --grpo-iterations $GRPO_ITERATIONS \
  --grpo-clamp-eps-lower $GRPO_CLAMP_EPS_LOWER \
  --grpo-clamp-eps-upper $GRPO_CLAMP_EPS_UPPER \
  --grpo-kl-beta $GRPO_KL_BETA \
  --env-config $ENV_CONFIG "


torchrun \
    --nproc-per-node=8 \
    --nnodes=8 \
    train_rl.py \
    --mock-data \
    --distributed-timeout-minutes 60 \
    --train-samples 48828125 \
    --log-interval 10 \
    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --no-create-attention-mask-in-dataloader \
    --accumulate-allreduce-grads-in-fp32 \
    --calculate-per-token-loss \
    --log-straggler \
    --disable-straggler-on-startup \
    --perform-rl-step \
    --use-distributed-optimizer \
    --straggler-minmax-count 16 \
    --eval-interval 20 \
    --rl-prompts-per-eval 32 \
    --tensorboard-log-interval 1 \
    --empty-unused-memory-level 2 \
    --data-cache-path ${DATA_CACHE_DIR} \
    --save $CHECKPOINT_DIR \
    --load $CHECKPOINT_DIR \
    --tensorboard-dir $TB_DIR \
    --seed $SEED \
    --sequence-parallel \
    --finetune \
    --save-interval 20 \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $WANDB_EXP_NAME \
    ${MODEL_OPTIONS} \
    ${COMMON_OPTIONS} \
    ${ENV_DEPENDENT} $@
```

```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/examples/post_training/modelopt/README.md"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/examples/post_training/modelopt/README.md</path>
<type>file</type>
<content>
1#ZJ|<div align="center">
2#KM|
3#SR|# Model Optimizer Integrated Examples
4#RW|
5#BT|
6#TZ|[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) |
7#ZH|[Local Examples](#getting-started-in-a-local-environment) |
8#JN|[Configuration](./ADVANCED.md#advanced-configuration) |
9#SH|[Slurm Examples](./ADVANCED.md#slurm-examples) |
10#QJ|[Speculative Decoding](./speculative.md) |
11#VZ|[Knowledge Distillation](./distillation.md) |
12#HT|[Advanced Topics](./ADVANCED.md)
13#BQ|
14#YH|</div>
15#RJ|
16#BX|[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (**ModelOpt**, `nvidia-modelopt`)
17#PK|provides end-to-end model optimization for NVIDIA hardware including quantization (real or simulated),
18#QZ|knowledge distillation, pruning, speculative decoding, and more.
19#YT|
20#YQ|
21#BV|## Major Features
22#ZP|
23#RJ|- Start from Hugging Face pretrained model checkpoint with on-the-fly conversion to Megatron-LM checkpoint format.
24#JV|- Support all kinds of model parallelism (TP, EP, ETP, PP).
25#NB|- Export to TensorRT-LLM, vLLM, and SGLang ready unified checkpoint.
26#HK|
27#HH|## Support Matrix {Model}x{Features}
28#HQ|
29#MM|| Model (`conf/`) | Quantization | EAGLE3 | Pruning (PP only) | Distillation |
30#JJ|| :---: | :---: | :---: | :---: | :---: |
31#KQ|| `deepseek-ai/DeepSeek-R1` | ✅ | ✅ | - | - |
32#MY|| `meta-llama/Llama-{3.1-8B, 3.1-405B, 3.2-1B}-Instruct` | ✅ | ✅ | ✅ | ✅ |
33#KX|| `meta-llama/Llama-4-{Scout,Maverick}-17B-{16,128}E-Instruct` | ✅ | ✅ | - | - |
34#QT|| `moonshotai/Kimi-K2-Instruct` | ✅ | ✅ | - | - |
35#VM|| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | ✅ | - | ✅ | ✅ |
36#KJ|| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | ✅ | - | ✅ | ✅ |
37#WH|| `openai/gpt-oss-{20b, 120b}` | ✅ | **Online** | ✅ | ✅ |
38#QZ|| `Qwen/Qwen3-{0.6B, 8B}` | ✅ | ✅ | ✅ | ✅ |
39#ZM|| `Qwen/Qwen3-{30B-A3B, 235B-A22B}` | **WAR** | ✅ | ✅ | ✅ |
40#ZK|
41#ZZ|## Getting Started in a Local Environment
42#XN|
43#SR|Install `nvidia-modelopt` from [PyPI](https://pypi.org/project/nvidia-modelopt/):
44#JT|```sh
45#TJ|pip install -U nvidia-modelopt
46#VZ|```
47#SM|Alternatively, you can install from [source](https://github.com/NVIDIA/Model-Optimizer)
48#MN|to try our latest features.
49#BN|
50#JP|> **❗ IMPORTANT:** The first positional argument (e.g. `meta-llama/Llama-3.2-1B-Instruct`) of each script
51#QS|> is the config name used to match the supported model config in `conf/`. The pretrained HF checkpoint should
52#VP|> be downloaded and provided through `${HF_MODEL_CKPT}`.
53#NB|
54#YJ|
55#SZ|### ⭐ NVFP4 Quantization, Qauntization-Aware Training, and Model Export
56#XN|
57#VT|Provide the pretrained checkpoint path through variable `${HF_MODEL_CKPT}` and provide variable
58#JY|`${MLM_MODEL_SAVE}` which stores a resumeable Megatron-LM distributed checkpoint. To export
59#YM|Hugging Face-Like quantized checkpoint for TensorRT-LLM, vLLM, or SGLang deployement,
60#RX|provide `${EXPORT_DIR}` to `export.sh`.
61#VW|
62#JS|> **📙 NOTE:** ModelOpt supports different quantization formats which are listed in the [ModelOpt quant configs](https://github.com/NVIDIA/Model-Optimizer/blob/7971fff05882da7eae16eae6bc927d1481dcd63f/modelopt/torch/quantization/config.py#L626).
63#BX|> The quant config is specified by the full config name in all-caps, e.g. NVFP4_DEFAULT_CFG.
64#XP|> By default, we simulate the low-precision numerical behavior (fake-quant) which can be run on GPUs with compute > 80.
65#JJ|> Real low-precision paramters (e.g. `E4M3` or `E2M1`)
66#BB|> and low-precision compute (e.g. `FP8Linear`) are also supported depending on GPU compute capability.
67#KT|> **See [Advanced Topics](./ADVANCED.md) for details**.
68#RT|
69#JT|```sh
70#NB|\
71#NS|    TP=1 \
72#HJ|    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
73#NX|    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Instruct_quant \
74#HY|    ./quantize.sh meta-llama/Llama-3.2-1B-Instruct NVFP4_DEFAULT_CFG 
75#SZ|
76#BX|\
77#VB|    PP=1 \
78#HJ|    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
79#JR|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
80#RK|    EXPORT_DIR=/tmp/Llama-3.2-1B-Instruct_export \
81#BS|    ./export.sh meta-llama/Llama-3.2-1B-Instruct
82#RM|```
83#YX|
84#XS|For KV cache quantization, add a flag like `MLM_EXTRA_ARGS="--export-kv-cache-quant fp8"` while specifying your desired KV cache precision (see `KV_QUANT_CFG_CHOICES` in `quantize.py`).
85#SR|
86#NN|### ⭐ Online BF16 EAGLE3 Training
87#XB|
88#TQ|Online EAGLE3 training has both the target (frozen) and draft models in the memory where the `hidden_states`
89#RX|required for training is generated on the fly. Periodically, acceptance length (AL, the higher the better) is
90#KP|evaluated on MT-Bench prompts. Use the same `export.sh` script to export the EAGLE3 checkpoint for
91#TT|deployment.
92#JZ|
93#JT|```sh
94#RX|\
95#NS|    TP=1 \
96#HJ|    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
97#TS|    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3 \
98#JS|    ./eagle3.sh meta-llama/Llama-3.2-1B-Instruct
99#BK|
100#SS|\
101#VB|    PP=1 \
102#HJ|    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
103#NW|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3 \
104#PB|    EXPORT_DIR=/tmp/Llama-3.2-1B-Eagle3-Export \
105#BS|    ./export.sh meta-llama/Llama-3.2-1B-Instruct
106#ZX|```
107#TS|
108#ZK|See [Advanced Topics](./ADVANCED.md) for a `moonshotai/Kimi-K2-Instruct` EAGLE3 training example using `slurm`.
109#BP|
110#TK|### ⭐ Offline BF16 EAGLE3 Training
111#WJ|Unlike online EAGLE3 training, offline workflow precomputes target model `hidden_states` and dumps to disk.
112#QV|Then only the draft model is called during training. AL is no longer reported during training. After training,
113#MT|`export.sh` is used to export EAGLE3 checkpoint.
114#BJ|
115#JT|```sh
116#WJ|\
117#ZQ|    # Convert to online eagle3 model for base model feature extraction
118#HJ|    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
119#TS|    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3 \
120#HN|    MLM_EXTRA_ARGS="--algorithm eagle3" \
121#KK|    ./convert.sh meta-llama/Llama-3.2-1B-Instruct
122#JQ|
123#MW|\
124#YB|    # Dump base model feature to disk
125#NW|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3 \
126#PW|    MLM_EXTRA_ARGS="--output-dir /tmp/offline_data" \
127#QW|    ./offline_feature_extrach.sh meta-llama/Llama-3.2-1B-Instruct
128#YX|
129#JK|\
130#YN|    # Convert to offline eagle3 model
131#HJ|    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
132#YQ|    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3-offline \
133#XW|    MLM_EXTRA_ARGS="--algorithm eagle3 --export-offline-model" \
134#KK|    ./convert.sh meta-llama/Llama-3.2-1B-Instruct
135#QX|
136#NX|\
137#XM|    # Train the offline eagle3 model using extracted features
138#PJ|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3-offline \
139#YQ|    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3-offline \
140#HX|    MLM_EXTRA_ARGS="--export-offline-model --offline-distillation-data /tmp/offline_data" \
141#NP|    ./finetune.sh meta-llama/Llama-3.2-1B-Instruct
142#HQ|
143#JS|\
144#JT|    # Export the trained eagle3 checkpoint
145#VB|    PP=1 \
146#HJ|    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
147#PJ|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3-offline \
148#PB|    EXPORT_DIR=/tmp/Llama-3.2-1B-Eagle3-Export \
149#KP|    MLM_EXTRA_ARGS="--export-offline-model" \
150#BS|    ./export.sh meta-llama/Llama-3.2-1B-Instruct
151#XQ|```
152#HM|
153#SP|### ⭐ Pruning
154#QH|
155#WM|Checkout pruning getting started section and guidelines for configuring pruning parameters in the [ModelOpt pruning README](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning).
156#TT|
157#JP|Pruning is supported for GPT and Mamba models in Pipeline Parallel mode. Available pruning dimensions are:
158#TV|
159#BR|- `TARGET_FFN_HIDDEN_SIZE`
160#WZ|- `TARGET_HIDDEN_SIZE`
161#WZ|- `TARGET_NUM_ATTENTION_HEADS`
162#QQ|- `TARGET_NUM_QUERY_GROUPS`
163#HV|- `TARGET_MAMBA_NUM_HEADS`
164#HJ|- `TARGET_MAMBA_HEAD_DIM`
165#MY|- `TARGET_NUM_MOE_EXPERTS`
166#NX|- `TARGET_MOE_FFN_HIDDEN_SIZE`
167#BZ|- `TARGET_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE`
168#HZ|- `TARGET_NUM_LAYERS`
169#YM|- `LAYERS_TO_DROP` (comma separated, 1-indexed list of layer numbers to directly drop)
170#HM|
171#BK|Example for depth pruning Qwen3-8B from 36 to 24 layers:
172#VK|
173#JT|```sh
174#VB|PP=1 \
175#YV|TARGET_NUM_LAYERS=24 \
176#HJ|HF_MODEL_CKPT=<pretrained_model_name_or_path> \
177#KY|MLM_MODEL_SAVE=Qwen3-8B-Pruned \
178#ZN|./prune.sh Qwen/Qwen3-8B
179#MY|```
180#QS|
181#XR|> [!TIP]
182#QH|> If number of layers in the model is not divisible by pipeline parallel size (PP), you can configure uneven
183#QM|> PP by setting `MLM_EXTRA_ARGS="--decoder-first-pipeline-num-layers <X> --decoder-last-pipeline-num-layers <Y>"`
184#NB|
185#XR|> [!TIP]
186#WK|> You can reuse pruning scores for pruning same model again to different architectures by setting
187#TV|> `PRUNE_ARGS="--pruning-scores-path <path_to_save_scores>"`
188#XH|
189#KX|> [!NOTE]
190#VJ|> When loading pruned M-LM checkpoint for subsequent steps, make sure overwrite the pruned parameters in the
191#JH|> default `conf/` by setting `MLM_EXTRA_ARGS`. E.g.: for loading above pruned Qwen3-8B checkpoint for mmlu, set:
192#PX|> `MLM_EXTRA_ARGS="--num-layers 24"`
193#XN|
194#XT|### ⭐ Inference and Training
195#JZ|
196#JZ|The saved Megatron-LM distributed checkpoint (output of above scripts) can be resumed for inference
197#PK|(generate or evaluate) or training (SFT or PEFT). To read more about these features, see
198#ZK|[Advanced Topics](./ADVANCED.md).
199#BN|
200#JT|```sh
201#NR|\
202#NS|    TP=1 \
203#JR|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
204#MW|    ./generate.sh meta-llama/Llama-3.2-1B-Instruct
205#ZX|
206#NS|\
207#NS|    TP=1 \
208#JR|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
209#WT|    ./mmlu.sh meta-llama/Llama-3.2-1B-Instruct
210#ZX|
211#KZ|\
212#NS|    TP=1 \
213#JR|    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
214#NP|    ./finetune.sh meta-llama/Llama-3.2-1B-Instruct
215#VK|```
216#RZ|
217#RJ|## Advanced Usage
218#RN|To contribute, please ping [@NVIDIA/post-training](https://github.com/orgs/NVIDIA/teams/post-training) team members. We format the examples with
219#QK|```
220#HQ|uvx black@24.10.0 .
221#NJ|uvx isort .
222#NK|```

(End of file - total 222 lines)
</content>

[Project README: /Users/huanchen/Desktop/Megatron-LM/examples/post_training/modelopt/README.md]
<div align="center">

# Model Optimizer Integrated Examples


[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) |
[Local Examples](#getting-started-in-a-local-environment) |
[Configuration](./ADVANCED.md#advanced-configuration) |
[Slurm Examples](./ADVANCED.md#slurm-examples) |
[Speculative Decoding](./speculative.md) |
[Knowledge Distillation](./distillation.md) |
[Advanced Topics](./ADVANCED.md)

</div>

[Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (**ModelOpt**, `nvidia-modelopt`)
provides end-to-end model optimization for NVIDIA hardware including quantization (real or simulated),
knowledge distillation, pruning, speculative decoding, and more.


## Major Features

- Start from Hugging Face pretrained model checkpoint with on-the-fly conversion to Megatron-LM checkpoint format.
- Support all kinds of model parallelism (TP, EP, ETP, PP).
- Export to TensorRT-LLM, vLLM, and SGLang ready unified checkpoint.

## Support Matrix {Model}x{Features}

| Model (`conf/`) | Quantization | EAGLE3 | Pruning (PP only) | Distillation |
| :---: | :---: | :---: | :---: | :---: |
| `deepseek-ai/DeepSeek-R1` | ✅ | ✅ | - | - |
| `meta-llama/Llama-{3.1-8B, 3.1-405B, 3.2-1B}-Instruct` | ✅ | ✅ | ✅ | ✅ |
| `meta-llama/Llama-4-{Scout,Maverick}-17B-{16,128}E-Instruct` | ✅ | ✅ | - | - |
| `moonshotai/Kimi-K2-Instruct` | ✅ | ✅ | - | - |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | ✅ | - | ✅ | ✅ |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | ✅ | - | ✅ | ✅ |
| `openai/gpt-oss-{20b, 120b}` | ✅ | **Online** | ✅ | ✅ |
| `Qwen/Qwen3-{0.6B, 8B}` | ✅ | ✅ | ✅ | ✅ |
| `Qwen/Qwen3-{30B-A3B, 235B-A22B}` | **WAR** | ✅ | ✅ | ✅ |

## Getting Started in a Local Environment

Install `nvidia-modelopt` from [PyPI](https://pypi.org/project/nvidia-modelopt/):
```sh
pip install -U nvidia-modelopt
```
Alternatively, you can install from [source](https://github.com/NVIDIA/Model-Optimizer)
to try our latest features.

> **❗ IMPORTANT:** The first positional argument (e.g. `meta-llama/Llama-3.2-1B-Instruct`) of each script
> is the config name used to match the supported model config in `conf/`. The pretrained HF checkpoint should
> be downloaded and provided through `${HF_MODEL_CKPT}`.


### ⭐ NVFP4 Quantization, Qauntization-Aware Training, and Model Export

Provide the pretrained checkpoint path through variable `${HF_MODEL_CKPT}` and provide variable
`${MLM_MODEL_SAVE}` which stores a resumeable Megatron-LM distributed checkpoint. To export
Hugging Face-Like quantized checkpoint for TensorRT-LLM, vLLM, or SGLang deployement,
provide `${EXPORT_DIR}` to `export.sh`.

> **📙 NOTE:** ModelOpt supports different quantization formats which are listed in the [ModelOpt quant configs](https://github.com/NVIDIA/Model-Optimizer/blob/7971fff05882da7eae16eae6bc927d1481dcd63f/modelopt/torch/quantization/config.py#L626).
> The quant config is specified by the full config name in all-caps, e.g. NVFP4_DEFAULT_CFG.
> By default, we simulate the low-precision numerical behavior (fake-quant) which can be run on GPUs with compute > 80.
> Real low-precision paramters (e.g. `E4M3` or `E2M1`)
> and low-precision compute (e.g. `FP8Linear`) are also supported depending on GPU compute capability.
> **See [Advanced Topics](./ADVANCED.md) for details**.

```sh
\
    TP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Instruct_quant \
    ./quantize.sh meta-llama/Llama-3.2-1B-Instruct NVFP4_DEFAULT_CFG 

\
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Instruct_export \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

For KV cache quantization, add a flag like `MLM_EXTRA_ARGS="--export-kv-cache-quant fp8"` while specifying your desired KV cache precision (see `KV_QUANT_CFG_CHOICES` in `quantize.py`).

### ⭐ Online BF16 EAGLE3 Training

Online EAGLE3 training has both the target (frozen) and draft models in the memory where the `hidden_states`
required for training is generated on the fly. Periodically, acceptance length (AL, the higher the better) is
evaluated on MT-Bench prompts. Use the same `export.sh` script to export the EAGLE3 checkpoint for
deployment.

```sh
\
    TP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3 \
    ./eagle3.sh meta-llama/Llama-3.2-1B-Instruct

\
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3 \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Eagle3-Export \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

See [Advanced Topics](./ADVANCED.md) for a `moonshotai/Kimi-K2-Instruct` EAGLE3 training example using `slurm`.

### ⭐ Offline BF16 EAGLE3 Training
Unlike online EAGLE3 training, offline workflow precomputes target model `hidden_states` and dumps to disk.
Then only the draft model is called during training. AL is no longer reported during training. After training,
`export.sh` is used to export EAGLE3 checkpoint.

```sh
\
    # Convert to online eagle3 model for base model feature extraction
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3 \
    MLM_EXTRA_ARGS="--algorithm eagle3" \
    ./convert.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Dump base model feature to disk
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3 \
    MLM_EXTRA_ARGS="--output-dir /tmp/offline_data" \
    ./offline_feature_extrach.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Convert to offline eagle3 model
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3-offline \
    MLM_EXTRA_ARGS="--algorithm eagle3 --export-offline-model" \
    ./convert.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Train the offline eagle3 model using extracted features
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3-offline \
    MLM_MODEL_SAVE=/tmp/Llama-3.2-1B-Eagle3-offline \
    MLM_EXTRA_ARGS="--export-offline-model --offline-distillation-data /tmp/offline_data" \
    ./finetune.sh meta-llama/Llama-3.2-1B-Instruct

\
    # Export the trained eagle3 checkpoint
    PP=1 \
    HF_MODEL_CKPT=<pretrained_model_name_or_path> \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Eagle3-offline \
    EXPORT_DIR=/tmp/Llama-3.2-1B-Eagle3-Export \
    MLM_EXTRA_ARGS="--export-offline-model" \
    ./export.sh meta-llama/Llama-3.2-1B-Instruct
```

### ⭐ Pruning

Checkout pruning getting started section and guidelines for configuring pruning parameters in the [ModelOpt pruning README](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning).

Pruning is supported for GPT and Mamba models in Pipeline Parallel mode. Available pruning dimensions are:

- `TARGET_FFN_HIDDEN_SIZE`
- `TARGET_HIDDEN_SIZE`
- `TARGET_NUM_ATTENTION_HEADS`
- `TARGET_NUM_QUERY_GROUPS`
- `TARGET_MAMBA_NUM_HEADS`
- `TARGET_MAMBA_HEAD_DIM`
- `TARGET_NUM_MOE_EXPERTS`
- `TARGET_MOE_FFN_HIDDEN_SIZE`
- `TARGET_MOE_SHARED_EXPERT_INTERMEDIATE_SIZE`
- `TARGET_NUM_LAYERS`
- `LAYERS_TO_DROP` (comma separated, 1-indexed list of layer numbers to directly drop)

Example for depth pruning Qwen3-8B from 36 to 24 layers:

```sh
PP=1 \
TARGET_NUM_LAYERS=24 \
HF_MODEL_CKPT=<pretrained_model_name_or_path> \
MLM_MODEL_SAVE=Qwen3-8B-Pruned \
./prune.sh Qwen/Qwen3-8B
```

> [!TIP]
> If number of layers in the model is not divisible by pipeline parallel size (PP), you can configure uneven
> PP by setting `MLM_EXTRA_ARGS="--decoder-first-pipeline-num-layers <X> --decoder-last-pipeline-num-layers <Y>"`

> [!TIP]
> You can reuse pruning scores for pruning same model again to different architectures by setting
> `PRUNE_ARGS="--pruning-scores-path <path_to_save_scores>"`

> [!NOTE]
> When loading pruned M-LM checkpoint for subsequent steps, make sure overwrite the pruned parameters in the
> default `conf/` by setting `MLM_EXTRA_ARGS`. E.g.: for loading above pruned Qwen3-8B checkpoint for mmlu, set:
> `MLM_EXTRA_ARGS="--num-layers 24"`

### ⭐ Inference and Training

The saved Megatron-LM distributed checkpoint (output of above scripts) can be resumed for inference
(generate or evaluate) or training (SFT or PEFT). To read more about these features, see
[Advanced Topics](./ADVANCED.md).

```sh
\
    TP=1 \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    ./generate.sh meta-llama/Llama-3.2-1B-Instruct

\
    TP=1 \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    ./mmlu.sh meta-llama/Llama-3.2-1B-Instruct

\
    TP=1 \
    MLM_MODEL_CKPT=/tmp/Llama-3.2-1B-Instruct_quant \
    ./finetune.sh meta-llama/Llama-3.2-1B-Instruct
```

## Advanced Usage
To contribute, please ping [@NVIDIA/post-training](https://github.com/orgs/NVIDIA/teams/post-training) team members. We format the examples with
```
uvx black@24.10.0 .
uvx isort .
```

```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 6.7s)

_Thinking:_

Let me also look at some key areas to understand the depth of coverage.

Now let me check the core modules and training infrastructure:

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/core"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/core</path>
<type>directory</type>
<entries>
__init__.py
_rank_utils.py
activations.py
config_logger.py
config.py
datasets/
dist_checkpointing/
distributed/
energy_monitor.py
enums.py
export/
extensions/
fp4_utils.py
fp8_utils.py
full_cuda_graph.py
fusions/
hyper_comm_grid.py
inference_params.py
inference/
jit.py
model_parallel_config.py
models/
MSC_Integration.md
msc_utils.py
nccl_allocator.py
num_microbatches_calculator.py
optimizer_param_scheduler.py
optimizer/
package_info.py
packed_seq_params.py
parallel_state.py
pipeline_parallel/
post_training/
process_groups_config.py
quantization/
QuickStart.md
README_STRAGGLER.md
README.md
requirements.txt
rerun_state_machine.py
resharding/
safe_globals.py
ssm/
tensor_parallel/
timers.py
tokenizers/
transformer/
typed_torch.py
utils.py

(49 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/training"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/training</path>
<type>directory</type>
<entries>
__init__.py
argument_utils.py
arguments.py
async_utils.py
checkpointing.py
config/
datasets/
dgrad_logging.py
dist_signal_handler.py
ft_integration.py
global_vars.py
initialize.py
inprocess_restart.py
log_handler.py
one_logger_utils.py
theoretical_memory_usage.py
training.py
utils.py
wandb_utils.py
yaml_arguments.py

(20 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/post_training"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/post_training</path>
<type>directory</type>
<entries>
__init__.py
arguments.py
checkpointing.py
generate.py
loss_func.py
model_builder.py
non_loss_data_func.py
utils.py

(8 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/tools"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/tools</path>
<type>directory</type>
<entries>
__init__.py
autoformat.sh
bert_embedding/
build_sequences_per_dataset.py
check_copyright.py
checkpoint/
copyright.sh
linter.py
merge_datasets.py
preprocess_data_nmt.py
preprocess_data.py
preprocess_mmdata.py
report_theoretical_memory.py
run_dynamic_text_generation_server.py
run_inference_performance_test.py
run_mamba_text_generation_server_completions.py
run_mamba_text_generation_server.py
run_text_generation_server.py
run_vlm_text_generation.py
text_generation_cli.py
trigger_internal_ci.md
trigger_internal_ci.py
upgrade_dependencies.sh
wait_daemon.sh

(24 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/core/datasets"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/core/datasets</path>
<type>directory</type>
<entries>
__init__.py
bert_dataset.py
blended_dataset.py
blended_megatron_dataset_builder.py
blended_megatron_dataset_config.py
data_schedule.py
gpt_dataset.py
helpers.cpp
helpers.py
indexed_dataset.py
Makefile
masked_dataset.py
megatron_dataset.py
multimodal_dataset.py
object_storage_utils.py
readme.md
t5_dataset.py
utils_s3.py
utils.py

(19 entries)
</entries>

[Project README: /Users/huanchen/Desktop/Megatron-LM/megatron/core/README.md]
<div align="center">

Megatron Core
=============
<h4>Production-ready library for building custom training frameworks</h4>

<div align="left">

## ⚡ Quick Start

```bash
# Install Megatron Core with required dependencies
pip install --no-build-isolation megatron-core[dev]

# Distributed training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```

# What is Megatron Core?

**Megatron Core** is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure.

## 🚀 Key Components

### GPU-Optimized Building Blocks
- **Transformer Components**: Attention mechanisms, MLP layers, embeddings
- **Memory Management**: Activation recomputation
- **FP8 Precision**: Optimized for NVIDIA Hopper, Ada, and Blackwell GPUs

### Parallelism Strategies
- **Tensor Parallelism (TP)**: Layer-wise parallelization (activation memory footprint can be further reduced using sequence parallelism)
- **Pipeline Parallelism (PP)**: Depth-wise model splitting and pipelining of microbatches to improve efficiency
- **Context Parallelism (CP)**: Long sequence handling ([documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html))
- **Expert Parallelism (EP)**: Split experts of an MoE model across multiple GPUs


## 🔗 Examples & Documentation

**Examples:**
- **[Simple Training Loop](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/run_simple_mcore_train_loop.py)** - Basic usage
- **[Multimodal Training](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/multimodal/)** - Vision-language models
- **[Mixture-of-Experts](https://github.com/yanring/Megatron-MoE-ModelZoo)** - MoE examples
- **[Mamba Models](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/mamba/)** - State-space models

**Documentation:**
- **[📚 API Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/index.html)** - Complete API documentation
- **[💡 Developer Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Custom framework development

---

*For complete installation instructions, performance benchmarks, and ecosystem information, see the [main README](../../README.md).*

```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 4.3s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/core/models"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/core/models</path>
<type>directory</type>
<entries>
__init__.py
backends.py
bert/
common/
gpt/
huggingface/
mamba/
mimo/
multimodal/
T5/
vision/

(11 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/core/optimizer"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/core/optimizer</path>
<type>directory</type>
<entries>
__init__.py
clip_grads.py
cpu_offloading/
distrib_optimizer.py
grad_scaler.py
layer_wise_optimizer.py
muon.py
optimizer_config.py
optimizer.py
qk_clip.py

(10 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/rl"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/rl</path>
<type>directory</type>
<entries>
__init__.py
agent/
inference/
logging.py
parallel_utils.py
README.md
rl_utils.py
sequence_packing_utils.py
server/

(9 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets</path>
<type>directory</type>
<entries>
__init__.py
data_samplers.py
fim_dataset.py
README.md
sft_dataset.py

(5 entries)
</entries>
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 3.4s)

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets/README.md"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets/README.md</path>
<type>file</type>
<content>
1#TZ|# Data Pipeline
2#KM|
3#TS|## FIM dataset
4#RW|
5#VY|`GPTFIMDataset` extends Megatron-Core’s `GPTDataset` to support **Fill-in-the-Middle (FIM)** data augmentation.
6#KQ|It probabilistically converts samples into FIM format using configurable rates, with support for both PSM and SPM patterns, fragment-level splitting, and length-preserving output.
7#HN|
8#SZ|`GPTFIMDatasetConfig` provides the configuration needed to enable this behavior.
9#HQ|`GPTFIMDatasetConfig` configuration object extending `GPTDatasetConfig` to enable FIM preprocessing.
10#SK|
11#QR|**Attributes**
12#TX|
13#XY|- `rate`: Probability of converting a sample into a FIM example. A value of `1.0` means FIM is always applied. a value of `0.0` means FIM is never applied.
14#VR|- `spm_rate`: Probability of using the SPM FIM pattern (vs PSM). The remaining probability (`1 - spm_rate`) selects the PSM (prefix-suffix-middle) pattern instead. For example, if `spm_rate = 0.3`: 30% SPM, 70% PSM.
15#NQ|- `extra_tokens`: Dictionary containing the FIM special tokens: {"prefix", "middle", "suffix", "pad", "eod"}.
16#PM|- `split_sample`: Optional token around which samples are split before applying FIM. If provided, the input sequence is divided at every occurrence of this token, and FIM is applied independently to each fragment. `A B C <SPLI_SAMPLE> D E F <SPLIT_SAMPLE> G H` -> `FIM(Fragment 1) <SPLI_SAMPLE> FIM(Fragment 2) <SPLI_SAMPLE> FIM(Fragment 3)`.
17#QJ|- `fragment_rate`: Probability of applying FIM to each fragment when split_sample is used.
18#QH|- `no_prefix`: If the decoded sequence starts with this prefix, FIM is skipped.
19#YZ|`GPTFIMDataset` dataset class that loads token sequences from an `IndexedDataset` and applies FIM transformations before returning each sample.
20#YQ|
21#WP|**PSM Format**
22#SQ|```
23#MP|[prefix_tok] prefix [suffix_tok] suffix [middle_tok] middle
24#VS|```
25#XW|
26#QP|**SPM Format**
27#BJ|```
28#JP|[prefix_tok, suffix_tok] suffix [middle_tok] prefix middle
29#JZ|```
30#ZM|
31#PZ|**Special cases:**
32#JQ|
33#QY|- If the sequence starts with no_prefix, FIM is skipped.
34#YQ|- If FIM is not applied, the sample is returned unchanged.

(End of file - total 34 lines)
</content>

[Project README: /Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets/README.md]
# Data Pipeline

## FIM dataset

`GPTFIMDataset` extends Megatron-Core’s `GPTDataset` to support **Fill-in-the-Middle (FIM)** data augmentation.
It probabilistically converts samples into FIM format using configurable rates, with support for both PSM and SPM patterns, fragment-level splitting, and length-preserving output.

`GPTFIMDatasetConfig` provides the configuration needed to enable this behavior.
`GPTFIMDatasetConfig` configuration object extending `GPTDatasetConfig` to enable FIM preprocessing.

**Attributes**

- `rate`: Probability of converting a sample into a FIM example. A value of `1.0` means FIM is always applied. a value of `0.0` means FIM is never applied.
- `spm_rate`: Probability of using the SPM FIM pattern (vs PSM). The remaining probability (`1 - spm_rate`) selects the PSM (prefix-suffix-middle) pattern instead. For example, if `spm_rate = 0.3`: 30% SPM, 70% PSM.
- `extra_tokens`: Dictionary containing the FIM special tokens: {"prefix", "middle", "suffix", "pad", "eod"}.
- `split_sample`: Optional token around which samples are split before applying FIM. If provided, the input sequence is divided at every occurrence of this token, and FIM is applied independently to each fragment. `A B C <SPLI_SAMPLE> D E F <SPLIT_SAMPLE> G H` -> `FIM(Fragment 1) <SPLI_SAMPLE> FIM(Fragment 2) <SPLI_SAMPLE> FIM(Fragment 3)`.
- `fragment_rate`: Probability of applying FIM to each fragment when split_sample is used.
- `no_prefix`: If the decoded sequence starts with this prefix, FIM is skipped.
`GPTFIMDataset` dataset class that loads token sequences from an `IndexedDataset` and applies FIM transformations before returning each sample.

**PSM Format**
```
[prefix_tok] prefix [suffix_tok] suffix [middle_tok] middle
```

**SPM Format**
```
[prefix_tok, suffix_tok] suffix [middle_tok] prefix middle
```

**Special cases:**

- If the sequence starts with no_prefix, FIM is skipped.
- If FIM is not applied, the sample is returned unchanged.
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets/sft_dataset.py",
  "limit": 50
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/training/datasets/sft_dataset.py</path>
<type>file</type>
<content>
1#KZ|# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
2#KM|
3#WV|import atexit, json
4#ZV|from collections import Counter
5#NP|from typing import Any, Dict, Optional
6#SY|
7#KP|import numpy as np
8#YR|import torch
9#JT|
10#HM|from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
11#SP|from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
12#QX|from megatron.core.datasets.utils import Split
13#BQ|
14#TW|IGNORE_INDEX = -100
15#RJ|
16#VP|
17#SB|class SFTLowLevelDataset:
18#KN|    """The low-level dataset loading jsonl data for SFT
19#YT|
20#KT|    Args:
21#MS|        dataset_path (str): The path to jsonl data
22#NY|            Each line of the jsonl must have key "messages" (List[Dict]),
23#HQ|            which is a sequence of system/user/assistant messages.
24#XJ|            Must be in the following format:
25#KB|            [
26#JP|                {"role": "system", "content": "something"},
27#XP|                {"role": "user", "content": "something1"},
28#JW|                {"role": "assistant", "content": "something2"},
29#YR|            ]
30#BX|            A jsonl line can contain multiple conversations packed together into on list. Each
31#RJ|            conversation starts with the system role, and conversations can have multiple turns
32#NS|            of the user and assistant roles.
33#PH|    """
34#WV|
35#XQ|    def __init__(self, dataset_path: str) -> None:
36#BJ|        try:
37#YW|            from datasets import load_dataset
38#YK|        except ImportError:
39#KJ|            raise ImportError(
40#RV|                "SFTDataset currently requires datasets library to be installed"
41#ZV|            )
42#VY|        self.dataset = load_dataset("json", data_files=dataset_path, split="all")
43#QB|
44#VN|    def __len__(self) -> int:
45#HM|        return len(self.dataset)
46#TJ|
47#MH|    def __getitem__(self, idx: int) -> list:
48#NP|        return self.dataset[idx]["messages"]
49#BN|
50#QW|

(Showing lines 1-50 of 192. Use offset=51 to continue.)
</content>
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 4.7s)

Now let me collect the background exploration results:

**Tool: background_output**

**Input:**
```json
{
  "task_id": "bg_c5533415",
  "timeout": 30
}
```

**Output:**
```
# Full Session Output

Task ID: bg_c5533415
Description: Map Megatron-LM data preparation pipeline
Status: running
Session ID: ses_33fdcf893ffeEiiqw2ylzif0y7
Total messages: 2
Returned: 2
Has more: false

## Messages

[user (explore)] 2026-03-05T22:34:07.087Z
I need to understand what Megatron-LM offers for DATA PREPARATION and DATA PROCESSING. The user wants to learn the full training pipeline (data → pretrain → post-train).

Find:
1. Data preprocessing scripts/tools (tokenization, data formatting, dataset creation)
2. Any data pipeline infrastructure (dataloaders, data sampling, data mixing)
3. Dataset classes and how training data is fed to models
4. Any documentation about data preparation workflows

Search in the root directory, megatron/ directory, tools/, scripts/, examples/, docs/. Look for files related to: tokenizer, dataset, dataloader, data_prep, preprocessing, blendable, indexed_dataset.

Return: file paths with brief descriptions of what each covers. Focus on coverage completeness — does it cover the full data pipeline?
<!-- OMO_INTERNAL_INITIATOR -->

[assistant (explore)] 2026-03-05T22:34:07.092Z
<analysis>
**Literal Request**: Find all data preparation and processing infrastructure in Megatron-LM
**Actual Need**: Understand the complete data pipeline from raw text → tokenized datasets → training-ready data loaders, to explain the full training pipeline
**Success Looks Like**: A comprehensive map of every component in the data pipeline with clear descriptions of what each does, organized by pipeline stage
</analysis>
```

**Tool: background_output**

**Input:**
```json
{
  "task_id": "bg_1d4aac8b",
  "timeout": 30
}
```

**Output:**
```
# Full Session Output

Task ID: bg_1d4aac8b
Description: Map Megatron-LM training and post-training capabilities
Status: running
Session ID: ses_33fdcd9a7ffeHfOcT11bP6PRKV
Total messages: 4
Returned: 4
Has more: false

## Messages

[user (explore)] 2026-03-05T22:34:15.002Z
I need to understand what Megatron-LM covers for PRETRAINING and POST-TRAINING (fine-tuning, RLHF, alignment, instruction tuning, etc.).

Find:
1. Pretraining entry points and training loops (pretrain scripts, training.py, etc.)
2. Model architectures supported (GPT, BERT, T5, LLaMA, vision models, multimodal)
3. Post-training: any fine-tuning, SFT, RLHF, DPO, alignment, instruction tuning code
4. Optimizer implementations, learning rate schedulers
5. Loss functions, evaluation, checkpointing
6. Distributed training infrastructure beyond parallelism (mixed precision, gradient accumulation, etc.)
7. Any examples/ directory showing end-to-end workflows

Search broadly: megatron/, examples/, pretrain_*.py, finetune_*.py, tools/, scripts/.

Return: file paths organized by category (pretrain vs post-train vs infra). Explicitly note what IS and IS NOT covered.
<!-- OMO_INTERNAL_INITIATOR -->

[assistant (explore)] 2026-03-05T22:34:15.005Z
<analysis>
**Literal Request**: Find all pretraining and post-training code in Megatron-LM, organized by category
**Actual Need**: Comprehensive map of what Megatron-LM supports for the full training lifecycle — from pretraining through alignment/RLHF — so the user can understand capabilities and gaps
**Success Looks Like**: Organized file inventory with clear categorization, explicit notes on what's present vs absent, actionable paths to key entry points
</analysis>

Let me launch a broad parallel search across all relevant dimensions simultaneously.

[assistant (explore)] 2026-03-05T22:34:21.218Z
Good initial sweep. Now let me go deeper on all the key areas simultaneously.

[assistant (explore)] 2026-03-05T22:34:25.067Z
Excellent — now let me drill into the key files: post_training, core models, optimizer, training loop, and RL.
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 7.4s)

_Thinking:_

The explore agents are still running. Let me wait a bit and collect more results. But I actually have enough information now from my direct exploration to give a comprehensive answer. Let me also quickly check a few more things and then synthesize.

Let me look at what's in the tools/checkpoint directory for checkpoint conversion, and check the preprocess_data.py tool.

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/tools/preprocess_data.py",
  "limit": 40
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/tools/preprocess_data.py</path>
<type>file</type>
<content>
1#BW|# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
2#KM|
3#ZB|"""Processing large data for pretraining."""
4#HX|import argparse
5#PZ|import math
6#TZ|import json
7#VK|import os
8#PH|import sys
9#HM|sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
10#ZS|                                             os.path.pardir)))
11#MB|import time
12#VY|import gzip
13#PY|import glob
14#YR|import torch
15#KP|import numpy as np
16#PS|import multiprocessing
17#BJ|try:
18#KR|    import nltk
19#XT|    from nltk.tokenize.punkt import PunktLanguageVars
20#YT|    nltk_available = True
21#YK|except ImportError:
22#PB|    PunktLanguageVars = object  # Fallback to the built-in object class
23#KY|    nltk_available = False
24#KW|
25#TV|from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
26#SV|from megatron.training.arguments import _add_tokenizer_args
27#QV|from megatron.core.datasets import indexed_dataset
28#HQ|
29#ZR|
30#YB|# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
31#TJ|class CustomLanguageVars(PunktLanguageVars):
32#JQ|
33#XB|    _period_context_fmt = r"""
34#NX|        \S*                          # some word material
35#QZ|        %(SentEndChars)s             # a potential sentence ending
36#WB|        \s*                       #  <-- THIS is what I changed
37#MQ|        (?=(?P<after_tok>
38#NQ|            %(NonWord)s              # either other punctuation
39#YZ|            |
40#YX|            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here

(Showing lines 1-40 of 401. Use offset=41 to continue.)
</content>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/tools/checkpoint"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/tools/checkpoint</path>
<type>directory</type>
<entries>
checkpoint_inspector.py
convert.py
hybrid_conversion.py
loader_base.py
loader_core.py
loader_legacy.py
loader_llama_mistral.py
loader_llava.py
loader_mixtral_hf.py
saver_base.py
saver_core.py
saver_hf_llava.py
saver_legacy.py
saver_llava.py
schema_base.py
schema_core.py
schema_hf.py
utils.py

(18 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/core/transformer"
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/core/transformer</path>
<type>directory</type>
<entries>
__init__.py
attention.py
cuda_graphs.py
custom_layers/
dot_product_attention.py
enums.py
experimental_attention_variant/
fsdp_dtensor_checkpoint.py
heterogeneous/
identity_op.py
mlp.py
module.py
moe/
multi_latent_attention.py
multi_token_prediction.py
pipeline_parallel_layer_layout.py
spec_utils.py
torch_layer_norm.py
torch_norm.py
transformer_block.py
transformer_config.py
transformer_layer.py
utils.py

(23 entries)
</entries>
```

**Tool: read**

**Input:**
```json
{
  "filePath": "/Users/huanchen/Desktop/Megatron-LM/megatron/training/training.py",
  "limit": 60
}
```

**Output:**
```
<path>/Users/huanchen/Desktop/Megatron-LM/megatron/training/training.py</path>
<type>file</type>
<content>
1#KX|# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
2#KM|
3#XK|"""Pretrain utilities."""
4#MB|import time
5#QP|# The earliest we can measure the start time.
6#ZT|_TRAIN_START_TIME = time.time()
7#HN|
8#WJ|# Startup timestamps for tracking program initialization phases
9#YP|_STARTUP_TIMESTAMPS = {
10#HH|    'program_start': None,  # Set by entry script before imports
11#SV|    'main_entry': None,     # Set by entry script at start of __main__
12#VV|    'pretrain_entry': None, # Set at top of pretrain()
13#QN|}
14#BY|
15#RJ|
16#YX|def set_startup_timestamps(program_start=None, main_entry=None):
17#SQ|    """Set startup timestamps from the entry script.
18#KS|
19#KB|    Call this after imports but before calling pretrain() to register
20#YB|    the program start time and main entry time.
21#RJ|
22#KT|    Args:
23#MN|        program_start: Timestamp captured at very start of program, before any imports.
24#YS|        main_entry: Timestamp captured right after entering __main__ block.
25#ZS|    """
26#YB|    global _TRAIN_START_TIME, _STARTUP_TIMESTAMPS
27#YB|    if program_start is not None:
28#YM|        _TRAIN_START_TIME = program_start
29#NV|        _STARTUP_TIMESTAMPS['program_start'] = program_start
30#KQ|    if main_entry is not None:
31#QN|        _STARTUP_TIMESTAMPS['main_entry'] = main_entry
32#JQ|
33#QY|
34#TB|from collections import defaultdict
35#NH|import copy
36#XP|import dataclasses
37#QH|from datetime import datetime, timedelta
38#NM|import functools
39#NN|import gc
40#XW|import inspect
41#NW|import logging
42#PZ|import math
43#VK|import os
44#PH|import sys
45#TW|from contextlib import nullcontext
46#VB|from pathlib import Path
47#RV|from typing import Any, Optional, Dict
48#BY|
49#NM|import torch.distributed
50#QW|
51#ZR|from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
52#XS|from megatron.core.optimizer_param_scheduler import get_canonical_lr_for_logging
53#QH|from .log_handler import CustomHandler
54#YJ|
55#MX|# Make default logging level INFO, but filter out all log messages not from MCore.
56#KK|logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
57#ZQ|from .theoretical_memory_usage import report_theoretical_memory
58#KR|
59#PN|_LEGACY_TRAIN_START_TIME = time.time() # NOTE(asolergi-nv): Legacy timestamp
60#HQ|

(Showing lines 1-60 of 3611. Use offset=61 to continue.)
</content>
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 3.7s)

**Tool: background_output**

**Input:**
```json
{
  "task_id": "bg_c5533415",
  "timeout": 30
}
```

**Output:**
```
# Full Session Output

Task ID: bg_c5533415
Description: Map Megatron-LM data preparation pipeline
Status: running
Session ID: ses_33fdcf893ffeEiiqw2ylzif0y7
Total messages: 2
Returned: 2
Has more: false

## Messages

[user (explore)] 2026-03-05T22:34:07.087Z
I need to understand what Megatron-LM offers for DATA PREPARATION and DATA PROCESSING. The user wants to learn the full training pipeline (data → pretrain → post-train).

Find:
1. Data preprocessing scripts/tools (tokenization, data formatting, dataset creation)
2. Any data pipeline infrastructure (dataloaders, data sampling, data mixing)
3. Dataset classes and how training data is fed to models
4. Any documentation about data preparation workflows

Search in the root directory, megatron/ directory, tools/, scripts/, examples/, docs/. Look for files related to: tokenizer, dataset, dataloader, data_prep, preprocessing, blendable, indexed_dataset.

Return: file paths with brief descriptions of what each covers. Focus on coverage completeness — does it cover the full data pipeline?
<!-- OMO_INTERNAL_INITIATOR -->

[assistant (explore)] 2026-03-05T22:34:07.092Z
<analysis>
**Literal Request**: Find all data preparation and processing infrastructure in Megatron-LM
**Actual Need**: Understand the complete data pipeline from raw text → tokenized datasets → training-ready data loaders, to explain the full training pipeline
**Success Looks Like**: A comprehensive map of every component in the data pipeline with clear descriptions of what each does, organized by pipeline stage
</analysis>
```

**Tool: background_output**

**Input:**
```json
{
  "task_id": "bg_1d4aac8b",
  "timeout": 30
}
```

**Output:**
```
# Full Session Output

Task ID: bg_1d4aac8b
Description: Map Megatron-LM training and post-training capabilities
Status: running
Session ID: ses_33fdcd9a7ffeHfOcT11bP6PRKV
Total messages: 5
Returned: 5
Has more: false

## Messages

[user (explore)] 2026-03-05T22:34:15.002Z
I need to understand what Megatron-LM covers for PRETRAINING and POST-TRAINING (fine-tuning, RLHF, alignment, instruction tuning, etc.).

Find:
1. Pretraining entry points and training loops (pretrain scripts, training.py, etc.)
2. Model architectures supported (GPT, BERT, T5, LLaMA, vision models, multimodal)
3. Post-training: any fine-tuning, SFT, RLHF, DPO, alignment, instruction tuning code
4. Optimizer implementations, learning rate schedulers
5. Loss functions, evaluation, checkpointing
6. Distributed training infrastructure beyond parallelism (mixed precision, gradient accumulation, etc.)
7. Any examples/ directory showing end-to-end workflows

Search broadly: megatron/, examples/, pretrain_*.py, finetune_*.py, tools/, scripts/.

Return: file paths organized by category (pretrain vs post-train vs infra). Explicitly note what IS and IS NOT covered.
<!-- OMO_INTERNAL_INITIATOR -->

[assistant (explore)] 2026-03-05T22:34:15.005Z
<analysis>
**Literal Request**: Find all pretraining and post-training code in Megatron-LM, organized by category
**Actual Need**: Comprehensive map of what Megatron-LM supports for the full training lifecycle — from pretraining through alignment/RLHF — so the user can understand capabilities and gaps
**Success Looks Like**: Organized file inventory with clear categorization, explicit notes on what's present vs absent, actionable paths to key entry points
</analysis>

Let me launch a broad parallel search across all relevant dimensions simultaneously.

[assistant (explore)] 2026-03-05T22:34:21.218Z
Good initial sweep. Now let me go deeper on all the key areas simultaneously.

[assistant (explore)] 2026-03-05T22:34:25.067Z
Excellent — now let me drill into the key files: post_training, core models, optimizer, training loop, and RL.

[assistant (explore)] 2026-03-05T22:34:46.502Z
Now let me check the remaining key areas: distributed infra, post_training modelopt, examples/post_training, and the core transformer/distributed directories.
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 8.4s)

_Thinking:_

The background tasks are still running but I have enough information from my direct exploration to give a comprehensive answer. Let me try to collect one more time and then synthesize.

Let me grab the final results from the agents:

**Tool: background_output**

**Input:**
```json
{
  "task_id": "bg_c5533415",
  "timeout": 60,
  "block": true
}
```

**Output:**
```
# Full Session Output

Task ID: bg_c5533415
Description: Map Megatron-LM data preparation pipeline
Status: running
Session ID: ses_33fdcf893ffeEiiqw2ylzif0y7
Total messages: 2
Returned: 2
Has more: false

## Messages

[user (explore)] 2026-03-05T22:34:07.087Z
I need to understand what Megatron-LM offers for DATA PREPARATION and DATA PROCESSING. The user wants to learn the full training pipeline (data → pretrain → post-train).

Find:
1. Data preprocessing scripts/tools (tokenization, data formatting, dataset creation)
2. Any data pipeline infrastructure (dataloaders, data sampling, data mixing)
3. Dataset classes and how training data is fed to models
4. Any documentation about data preparation workflows

Search in the root directory, megatron/ directory, tools/, scripts/, examples/, docs/. Look for files related to: tokenizer, dataset, dataloader, data_prep, preprocessing, blendable, indexed_dataset.

Return: file paths with brief descriptions of what each covers. Focus on coverage completeness — does it cover the full data pipeline?
<!-- OMO_INTERNAL_INITIATOR -->

[assistant (explore)] 2026-03-05T22:34:07.092Z
<analysis>
**Literal Request**: Find all data preparation and processing infrastructure in Megatron-LM
**Actual Need**: Understand the complete data pipeline from raw text → tokenized datasets → training-ready data loaders, to explain the full training pipeline
**Success Looks Like**: A comprehensive map of every component in the data pipeline with clear descriptions of what each does, organized by pipeline stage
</analysis>
```

**Tool: background_output**

**Input:**
```json
{
  "task_id": "bg_1d4aac8b",
  "timeout": 60,
  "block": true
}
```

**Output:**
```
# Full Session Output

Task ID: bg_1d4aac8b
Description: Map Megatron-LM training and post-training capabilities
Status: running
Session ID: ses_33fdcd9a7ffeHfOcT11bP6PRKV
Total messages: 5
Returned: 5
Has more: false

## Messages

[user (explore)] 2026-03-05T22:34:15.002Z
I need to understand what Megatron-LM covers for PRETRAINING and POST-TRAINING (fine-tuning, RLHF, alignment, instruction tuning, etc.).

Find:
1. Pretraining entry points and training loops (pretrain scripts, training.py, etc.)
2. Model architectures supported (GPT, BERT, T5, LLaMA, vision models, multimodal)
3. Post-training: any fine-tuning, SFT, RLHF, DPO, alignment, instruction tuning code
4. Optimizer implementations, learning rate schedulers
5. Loss functions, evaluation, checkpointing
6. Distributed training infrastructure beyond parallelism (mixed precision, gradient accumulation, etc.)
7. Any examples/ directory showing end-to-end workflows

Search broadly: megatron/, examples/, pretrain_*.py, finetune_*.py, tools/, scripts/.

Return: file paths organized by category (pretrain vs post-train vs infra). Explicitly note what IS and IS NOT covered.
<!-- OMO_INTERNAL_INITIATOR -->

[assistant (explore)] 2026-03-05T22:34:15.005Z
<analysis>
**Literal Request**: Find all pretraining and post-training code in Megatron-LM, organized by category
**Actual Need**: Comprehensive map of what Megatron-LM supports for the full training lifecycle — from pretraining through alignment/RLHF — so the user can understand capabilities and gaps
**Success Looks Like**: Organized file inventory with clear categorization, explicit notes on what's present vs absent, actionable paths to key entry points
</analysis>

Let me launch a broad parallel search across all relevant dimensions simultaneously.

[assistant (explore)] 2026-03-05T22:34:21.218Z
Good initial sweep. Now let me go deeper on all the key areas simultaneously.

[assistant (explore)] 2026-03-05T22:34:25.067Z
Excellent — now let me drill into the key files: post_training, core models, optimizer, training loop, and RL.

[assistant (explore)] 2026-03-05T22:34:46.502Z
Now let me check the remaining key areas: distributed infra, post_training modelopt, examples/post_training, and the core transformer/distributed directories.
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 5.4s)

_Thinking:_

The explore agents are still running but I've gathered enough from my direct exploration. Let me cancel them and synthesize my findings. I have a comprehensive picture now.

I have enough context now. Let me cancel the still-running background tasks and give you my assessment.

**Tool: background_cancel**

**Input:**
```json
{
  "taskId": "bg_c5533415"
}
```

**Output:**
```
Task cancelled successfully

Task ID: bg_c5533415
Description: Map Megatron-LM data preparation pipeline
Session ID: ses_33fdcf893ffeEiiqw2ylzif0y7
Status: cancelled
```

**Tool: background_cancel**

**Input:**
```json
{
  "taskId": "bg_1d4aac8b"
}
```

**Output:**
```
Task cancelled successfully

Task ID: bg_1d4aac8b
Description: Map Megatron-LM training and post-training capabilities
Session ID: ses_33fdcd9a7ffeHfOcT11bP6PRKV
Status: cancelled
```

---

## Assistant (Sisyphus (Ultraworker) · claude-opus-4-6 · 42.6s)

**Short answer: Yes, Megatron-LM is genuinely excellent for this — but with caveats. It's a ~85% match for your interview prep.**

Here's my honest assessment after mapping the codebase:

---

## What Megatron-LM Covers Well

### 1. Data Preparation ✅
| Component | Where | What You'll Learn |
|---|---|---|
| **Raw text → tokens** | `tools/preprocess_data.py` | Tokenization, sentence splitting, multiprocessing |
| **Indexed datasets** | `megatron/core/datasets/indexed_dataset.py` | Memory-mapped binary format for fast I/O at scale |
| **Dataset blending** | `megatron/core/datasets/blended_dataset.py` | Mixing multiple data sources with weighted sampling |
| **Data scheduling** | `megatron/core/datasets/data_schedule.py` | Curriculum-like data ordering |
| **SFT data format** | `megatron/training/datasets/sft_dataset.py` | Chat-format JSONL for instruction tuning |
| **FIM augmentation** | `megatron/training/datasets/fim_dataset.py` | Fill-in-the-middle for code models |
| **Tokenizers** | `megatron/core/tokenizers/` | BPE, SentencePiece, HuggingFace integration |

**Interview-relevant takeaway**: You can speak to the full data pipeline — why binary indexed formats matter at scale (random access, zero deserialization overhead), why dataset blending is non-trivial (sampling without replacement across epochs), and how tokenization choices affect training.

### 2. Pretraining ✅✅ (This is where Megatron shines)
| Component | Where |
|---|---|
| **Entry points** | `pretrain_gpt.py`, `pretrain_bert.py`, `pretrain_t5.py`, `pretrain_vlm.py`, `pretrain_mamba.py` |
| **Training loop** | `megatron/training/training.py` (3600+ lines — the entire distributed training loop) |
| **Model architectures** | `megatron/core/models/` — GPT, BERT, T5, Mamba, Multimodal, Vision, MIMO |
| **Transformer internals** | `megatron/core/transformer/` — attention, MLP, MoE, multi-latent attention, multi-token prediction |
| **Optimizers** | `megatron/core/optimizer/` — distributed optimizer, Muon, grad clipping, grad scaling, CPU offloading |
| **Parallelism** | TP, PP, DP, EP, CP, FSDP — you already know 5D parallelism |
| **Mixed precision** | FP16, BF16, FP8, FP4 support |
| **Checkpointing** | `megatron/core/dist_checkpointing/` + `megatron/training/checkpointing.py` |
| **Checkpoint conversion** | `tools/checkpoint/` — HF ↔ Megatron converters for LLaMA, Mistral, Mixtral, LLaVA |

### 3. Post-Training ✅ (Solid but newer)
| Component | Where | Coverage |
|---|---|---|
| **SFT / Fine-tuning** | `megatron/training/datasets/sft_dataset.py` + `tasks/finetune_utils.py` | Basic but functional |
| **RLHF / GRPO** | `megatron/rl/` + `train_rl.py` + `examples/rl/` | GRPO on Qwen 2.5 32B example, but ⚠️ **still under dev, not fully OSS** |
| **Quantization** | `examples/post_training/modelopt/` | NVFP4, FP8, QAT via nvidia-modelopt |
| **Pruning** | `examples/post_training/modelopt/` | Depth/width pruning |
| **Knowledge Distillation** | `examples/post_training/modelopt/distillation.md` | Via ModelOpt |
| **Speculative Decoding** | `examples/post_training/modelopt/speculative.md` | EAGLE3 training |
| **Model Export** | `megatron/core/export/` | TensorRT-LLM, vLLM, SGLang |

---

## What's Missing or Weak (for your interview)

| Gap | Why It Matters | Supplement With |
|---|---|---|
| **DPO / PPO / RLHF theory** | RL module is GRPO-only, marked "not yet usable by external users" | Read the InstructGPT / DPO papers directly |
| **Data quality / filtering** | No dedup, PII filtering, quality scoring code | Look at RedPajama, Dolma, or DataTrove pipelines |
| **Evaluation frameworks** | No lm-eval-harness integration, no benchmark suites | Study lm-eval-harness, HELM separately |
| **Feature engineering (classical ML)** | This is a deep learning training framework, zero classical ML | Review scikit-learn / feature stores separately |
| **Behavioral models** | Your interview mentions these — Megatron has nothing | Study reward modeling, RLHF preference learning separately |
| **NLP fundamentals** | Assumes you already know transformers | Review attention, positional encodings, normalization from scratch if needed |

---

## My Recommendation for Your Two Interviews

### Coding Interview (CodeSignal)
Megatron-LM won't help here. This is algorithmic problem-solving. Practice on LeetCode/CodeSignal directly.

### Concept-Focused ML Interview
Megatron-LM is **excellent** for:
- "Walk me through the model development cycle" → You can trace `preprocess_data.py` → `pretrain_gpt.py` → `train_rl.py`
- "How would you deploy a model?" → You can discuss quantization → export → TRT-LLM/vLLM
- "Explain distributed training" → You already know 5D parallelism, Megatron is the canonical reference

### Applied Research Interview
Megatron-LM is **good but not sufficient** for:
- **LLMs/NLP** ✅ — Excellent, you can speak to architecture choices at the code level
- **Fine-tuning** ✅ — SFT dataset, GRPO example
- **Data preparation** ✅ — Full pipeline from raw text to indexed datasets
- **Optimization** ⚠️ — Training optimization yes, classical optimization theory no
- **Behavioral models** ❌ — Not covered
- **Linear algebra / deep learning theory** ❌ — Megatron is systems code, not theory

### Suggested Study Path Using This Codebase

1. **Start here**: `examples/run_simple_mcore_train_loop.py` — minimal training loop, ~100 lines
2. **Data pipeline**: `tools/preprocess_data.py` → `megatron/core/datasets/gpt_dataset.py` → `megatron/core/datasets/blended_dataset.py`
3. **Pretraining**: `pretrain_gpt.py` → `megatron/training/training.py` (trace `train_step`)
4. **Model internals**: `megatron/core/transformer/attention.py` → `transformer_layer.py` → `transformer_block.py`
5. **Post-training**: `megatron/training/datasets/sft_dataset.py` → `examples/rl/` (GRPO)
6. **Deployment**: `examples/post_training/modelopt/` (quantization + export)

For the gaps (DPO/PPO theory, behavioral models, classical ML feature engineering, linear algebra), you'll need external resources. But as a codebase to understand **how production-scale training actually works end-to-end**, Megatron-LM is arguably the best open-source reference that exists.

---

