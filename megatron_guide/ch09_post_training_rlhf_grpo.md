# Chapter 9 — Post-Training: RLHF & GRPO

> **Prerequisites**: Chapter 8 (SFT & LoRA). This chapter builds on the SFT-tuned model and turns it into a policy that improves from reward signals.

---

## 9.1 Why RL After SFT?

SFT teaches a model *how* to format answers — follow instructions, use a chat template, write code inside fences. But SFT can't teach a model to **reason better**, because the training signal is binary: did you match the reference token, yes or no?

Reinforcement Learning from Human Feedback (RLHF) replaces that binary signal with a **scalar reward** — a number that says "this answer is 8/10 good." The model can now learn that some correct-looking answers are *better* than others.

```
SFT:    "Match this exact reference"        → imitates teacher
RLHF:   "Here's a score for your attempt"   → explores and improves
```

### The RLHF Timeline

| Year | Method | Key Idea |
|------|--------|----------|
| 2020 | PPO-based RLHF | Train reward model → PPO policy optimization |
| 2023 | DPO | Skip reward model, train directly on preference pairs |
| 2024 | GRPO (DeepSeek) | Group-relative scoring, no reward model needed for verifiable tasks |
| 2025 | GRPO + Online RL | Continuous rollout collection with live inference |

Megatron-LM implements **GRPO** as its primary RL algorithm, with an architecture designed for massive-scale online RL training.

---

## 9.2 GRPO vs PPO vs DPO — The Big Picture

### PPO (Proximal Policy Optimization)

The classic RLHF approach:

1. **Train a reward model** on human preference pairs: (prompt, chosen, rejected)
2. **Generate rollouts** from the policy (the LLM)
3. **Score them** with the reward model
4. **Update the policy** using the PPO objective with a clipped surrogate loss

PPO requires four models in memory simultaneously:
- **Policy** (the model being trained)
- **Reference policy** (frozen copy to compute KL penalty)
- **Reward model** (scores outputs)
- **Value model / critic** (estimates expected reward)

This is extremely expensive in memory.

### DPO (Direct Preference Optimization)

DPO reformulates the reward model into the policy optimization:

```
L_DPO = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
```

- No reward model or value model needed (just policy + reference)
- Offline: trains on pre-collected preference pairs
- Simpler but limited by the quality of the preference data

### GRPO (Group Relative Policy Optimization)

GRPO is what DeepSeek introduced and what Megatron implements:

1. **Sample a group** of K responses for each prompt
2. **Score each response** with a verifiable reward (e.g., math answer correct? code passes tests?)
3. **Compute advantages** relative to the group mean — no value model needed
4. **Update the policy** using a PPO-like clipped objective

```
For prompt x, sample K responses: {y₁, y₂, ..., yₖ}
Get rewards: {r₁, r₂, ..., rₖ}
Advantage(yᵢ) = (rᵢ - mean(r)) / (std(r) + ε)
```

**Why GRPO is powerful:**
- No reward model training (uses verifiable rewards or rule-based scoring)
- No value model (group statistics replace the critic)
- Only needs policy + reference (2 models, not 4)
- Natural fit for math, code, and reasoning tasks where correctness is verifiable

---

## 9.3 The GRPO Math — Interview Deep Dive

### The GRPO Loss Function

The per-token loss for GRPO is:

```
L(θ) = -min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A) + β · KL(π_ref || π_θ) - α · H(π_θ)
```

Where:
- **r(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)** — the probability ratio (how much the policy changed)
- **A** — the group-relative advantage
- **clip(r(θ), 1-ε, 1+ε)** — clamp the ratio to prevent too-large updates (from PPO)
- **β · KL(π_ref || π_θ)** — KL penalty to stay close to the reference policy
- **α · H(π_θ)** — entropy bonus to encourage exploration

### Advantage Computation

```python
# From rl_utils.py: calculate_grpo_advantages()
rewards = np.array(rewards)        # shape: [num_groups, group_size]
mean = rewards.mean(axis=1)        # per-group mean
std = rewards.std(axis=1)          # per-group std
advantages = (rewards - mean) / (std + 1e-4)
```

Key insight: advantages are **relative within each group**. A response with reward 0.8 gets positive advantage if the group mean is 0.6, but negative advantage if the group mean is 0.9. This is the "Group Relative" in GRPO.

### The KL Penalty

Megatron uses the **unbiased KL estimator** (Schulman approximation):

```python
# From calculate_grpo_loss()
ref_diff = ref_logprobs - current_logprobs  # log(π_ref/π_θ)
kl_term = ref_diff.exp() - ref_diff - 1     # ≈ KL(π_ref || π_θ), unbiased
```

This is better than the naive `KL = Σ π_ref · log(π_ref/π_θ)` because:
1. It's computed per-token (doesn't need the full distribution)
2. It's always non-negative
3. It has lower variance

### The Clipped Surrogate

```python
# From calculate_grpo_loss()
ratios = (current_logprobs - old_logprobs).exp()       # π_θ / π_old
clamped_ratios = ratios.clamp(1 - eps_lower, 1 + eps_upper)

loss = -min(ratios * advantages, clamped_ratios * advantages)
      + kl_beta * kl_term
      - entropy_weight * entropy_term
```

The `min` operation is the PPO clip trick:
- If advantage > 0 (good response): don't let the ratio go above 1+ε (don't over-exploit)
- If advantage < 0 (bad response): don't let the ratio go below 1-ε (don't over-penalize)

### Importance Sampling Correction

Because inference might happen with a slightly stale policy (the model was updated since the rollout was generated), Megatron optionally applies importance sampling correction:

```python
# From calculate_grpo_loss()
if inference_logprobs is not None:
    is_weights = (old_logprobs - inference_logprobs).exp()  # π_old / π_inference
    if is_truncation_coef is not None:
        is_weights = torch.min(is_weights, truncation_coef)

loss = -is_weights * min(ratios * A, clamped * A) + ...
```

This corrects for the distribution shift between the inference-time policy and the training-time policy.

---

## 9.4 Megatron-RL Architecture — The Agent/Environment Pattern

Megatron-RL is architected as a **decoupled agent/environment system**:

```
┌─────────────────────────────────────────────────┐
│                   train_rl.py                    │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │ Inference │───▶│  Agent/  │───▶│  Trainer  │  │
│  │ Interface │    │   Env    │    │  (GRPO)   │  │
│  └──────────┘    └──────────┘    └───────────┘  │
│       ▲               │               │         │
│       │          Rollouts +        Gradient      │
│       │          Rewards           Updates       │
│       │               │               │         │
│  ┌──────────┐         │          ┌─────────┐    │
│  │   Model   │◀───────┘          │Optimizer│    │
│  │(inference)│                   └─────────┘    │
│  └──────────┘                                   │
└─────────────────────────────────────────────────┘
```

### Component Breakdown

**1. InferenceInterface** (`megatron/rl/inference/`)
- Wraps the model for text generation
- Supports multiple backends: Megatron local, OpenAI-compatible API, HuggingFace
- The `MegatronLocal` backend runs inference in-process with CUDA graphs

**2. Agent / Environment** (`megatron/rl/agent/`)
- Defines *how* to interact with the model and *how* to score responses
- `GroupedRolloutGenerator`: generates K responses per prompt (for GRPO)
- `WeightedMultiTask`: manages multiple environments with weighted sampling
- Returns `Rollout` or `TokenRollout` objects with trajectories + rewards

**3. Trainer** (`train_rl.py` + `rl_utils.py`)
- Runs the GRPO update loop: collect rollouts → compute logprobs → compute loss → gradient update
- Reuses Megatron's `pretrain()` function with a custom `forward_step`

### Key Data Structures

```python
class TokenRollout:
    trajectory: list[list[int]]     # Token IDs (list of turns)
    reward: float                    # Scalar reward for this response
    generation_mask: list[list[bool]]  # Which tokens were generated (vs prompt)
    logprobs: list[list[float]]      # Log-probs from inference engine
    env_id: str                      # Which environment produced this
    policy_staleness: list[list[int]]  # How stale the inference policy was
    
class GroupedRolloutRequest:
    num_groups: int                  # How many prompts
    rollouts_per_group: int          # K responses per prompt
    inference_interface: InferenceInterface
    generation_args: dict            # temperature, top_p, max_tokens
```

---

## 9.5 The GRPO Training Loop — Step by Step

### Entry Point: `train_rl.py`

```python
if __name__ == "__main__":
    pretrain(
        None,                    # No standard dataset (RL generates its own data)
        model_provider,          # Same GPT model as pretraining
        ModelType.encoder_or_decoder,
        forward_step,            # Custom forward_step for GRPO
        args_defaults={},
        extra_args_provider=add_inference_args,
    )
```

RL training reuses Megatron's `pretrain()` but passes `None` for the dataset provider and a custom `forward_step`.

### The Training Loop (Detailed)

```
For each iteration:
│
├─ 1. COLLECT ROLLOUTS (if needed)
│   ├─ Check if we need new data (every grpo_iterations × batches_per_collection)
│   ├─ Enter inference mode (model.eval(), enable CUDA graphs)
│   ├─ Generate K responses per prompt via the Agent
│   ├─ Score responses (environment returns rewards)
│   ├─ Broadcast rollouts from rank 0 to all ranks
│   └─ Exit inference mode (restore training state)
│
├─ 2. PREPARE DATA FOR UPDATE
│   ├─ Compute group statistics and advantages
│   ├─ Split rollouts across data-parallel ranks
│   ├─ Tokenize and pad trajectories
│   ├─ Compute π_old logprobs (forward pass, no grad)
│   ├─ Compute π_ref logprobs (load ref weights, forward, restore current weights)
│   ├─ Optionally pack sequences for efficiency
│   └─ Build DataLoader for the update step
│
├─ 3. GRADIENT UPDATE (standard Megatron train_step)
│   ├─ forward_step: run π_θ forward pass, get current logprobs
│   ├─ calculate_grpo_loss: compute clipped surrogate + KL + entropy
│   ├─ loss_func: aggregate loss, report metrics (KL, ratios, truncation rates)
│   └─ Backward pass + optimizer step (same as pretraining)
│
└─ 4. LOGGING & EVALUATION
    ├─ Log: mean reward, KL term, ratio stats, truncation rates, entropy
    ├─ Periodically run evaluation (separate prompts, no gradient)
    └─ Checkpoint if needed
```

### Step 1: Rollout Collection (`get_environment_rollouts`)

```python
def get_environment_rollouts(model, inference_model, optimizer, n_prompts, samples_per_group):
    # Optionally offload optimizer to CPU to free GPU memory for inference
    if args.rl_offload_optimizer_during_inference:
        optimizer.offload_to_cpu()
    
    # If separate inference model, copy weights from training model
    if inference_model is not None:
        swap_model_weights(model, inference_model, args.refit_method)
    
    # Enter inference context: eval mode, CUDA graphs, inference server
    with megatron_rl_inference_mode(inference_model, optimizer, ...) as inference_interface:
        # Rank 0 collects rollouts via the agent
        agent = get_agent(args)  # WeightedMultiTask from YAML config
        request = GroupedRolloutRequest(
            num_groups=n_prompts,
            rollouts_per_group=samples_per_group,
            inference_interface=inference_interface,
            generation_args={'temperature': 0.7, 'max_tokens': 4096, ...}
        )
        rollouts = [await anext(rollout_generator) for _ in range(n_prompts)]
    
    # Broadcast rollouts to all ranks
    torch.distributed.broadcast_object_list(rollouts, src=0)
    return rollouts
```

Key engineering decisions:
- **Optimizer offloading**: during inference, optimizer states are moved to CPU to free GPU memory
- **Separate inference model**: allows training and inference to have different configurations (e.g., inference uses CUDA graphs, training doesn't)
- **Weight swapping**: `swap_model_weights()` efficiently copies trained weights to the inference model via refitting (no full copy)

### Step 2: Data Preparation (`prepare_data_for_update`)

This is the most complex step:

```python
def prepare_data_for_update(model, ref_state_dict, rollouts, tokenizer, ...):
    # 1. Compute group advantages
    group_stats = compute_group_stats(rollouts, tokenizer, seq_length)
    advantages = calculate_grpo_advantages(group_stats.rewards, group_stats.num_turns)
    
    # 2. Split across DP ranks
    rollouts = rollouts[dp_start:dp_end]
    
    # 3. Tokenize and pad trajectories
    trajs, generation_masks, inference_logprobs = prepare_trajectories(rollouts, tokenizer, seq_length)
    
    # 4. Compute π_old logprobs
    old_logprobs = compute_logprobs_batch(model, data_loader, ...)
    
    # 5. Compute π_ref logprobs (temporarily load reference weights)
    cur_state = model.state_dict()          # save current weights
    model.load_state_dict(ref_state_dict)   # load reference weights
    ref_logprobs = compute_logprobs_batch(model, data_loader, ...)
    model.load_state_dict(cur_state)        # restore current weights
    
    # 6. Build cycled DataLoader
    return RerunDataIterator(itertools.cycle(loader)), group_stats, example_groups
```

The reference logprob computation is notable: instead of keeping a full reference model in memory, Megatron **temporarily swaps the model weights**, computes logprobs, and swaps back. This halves the memory requirement compared to keeping two full models.

### Step 3: The GRPO Forward Step

```python
def forward_step(data_iterator, model, loss_only=False):
    # Load batch: tokens, advantages, old_logprobs, loss_mask, position_ids, ref_logprobs
    batch_data = next(data_iterator)
    
    # Get current policy logprobs (WITH gradient)
    current_logprobs = get_logprobs(model, tokens, position_ids, no_grad=False)
    
    # Compute GRPO loss
    loss, kl_term, ratios, entropy_term, trunc_above, trunc_below = calculate_grpo_loss(
        current_logprobs=current_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        advantages=advantages,
        clamp_eps_lower=args.grpo_clamp_eps_lower,
        clamp_eps_upper=args.grpo_clamp_eps_upper,
        kl_beta=args.grpo_kl_beta,
        entropy_weight=args.grpo_entropy_term_weight,
    )
    
    return loss, partial(loss_func, loss_mask, kl_term, ratios, ...)
```

### Step 4: Loss Aggregation

```python
def loss_func(loss_mask, kl_term, ratios, entropy_term, ..., output_tensor):
    losses = output_tensor.float()
    total_tokens = loss_mask.sum()
    loss = (losses * loss_mask).sum()
    
    # NaN detection and spiky loss detection
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(loss, torch.isnan, ...)
    
    # Report metrics: KL, ratios, entropy, truncation rates
    return (loss * args.context_parallel_size, total_tokens, {
        'lm loss': loss,
        'rl/kl_term': masked_kl,
        'rl/pi_over_pi_old': masked_ratios,
        'rl/entropy_term': masked_entropy,
        'rl/truncated_from_above': ...,
        'rl/truncated_from_below': ...,
    })
```

---

## 9.6 Sequence Packing in RL

Just like SFT (Chapter 8), RL training benefits from sequence packing — fitting multiple variable-length rollouts into fixed-size bins to avoid wasted padding.

```
Without packing:                    With packing:
[prompt|response|PAD|PAD|PAD]      [prompt1|resp1|prompt2|resp2]
[prompt|resp|PAD|PAD|PAD|PAD]      [prompt3|response3|prompt4..]
                  ↑ wasted           ↑ efficient
```

The packing pipeline:
1. **`pack_all_trajectories()`**: uses bin-packing algorithms to fit multiple sequences into bins of size `seq_length`
2. **`cu_seqlens`** in `PackedSeqParams`: tells FlashAttention where sequence boundaries are (same mechanism as SFT)
3. **Advantage mapping**: each sequence in a packed bin gets its advantage mapped to the correct token positions:

```python
# From calculate_grpo_loss, packed path:
for seq_idx, (start, seq_len) in enumerate(zip(seq_starts, seq_lengths)):
    end = min(start + seq_len - 1, bin_size)
    packed_advantages[0, start:end] = advantages[seq_idx].item()
```

---

## 9.7 Memory Management Tricks

RL training is uniquely memory-hungry because it needs to alternate between inference (generating responses) and training (gradient updates). Megatron uses several tricks:

### 1. Optimizer Offloading During Inference

```python
# Before inference
optimizer.offload_to_cpu()       # Move Adam states to CPU (frees ~2x model size)
model.offload_grad_buffers()     # Free gradient buffers too

# After inference, before training
model.restore_grad_buffers()
optimizer.restore_from_cpu()
```

### 2. Reference Model Weight Swapping

Instead of keeping a full reference model in GPU memory:
```python
# Save current weights → load reference → compute logprobs → restore current
cur_state = {k: v.cpu() for k, v in model.state_dict().items()}
model.load_state_dict(ref_state_dict)    # reference model in GPU
ref_logprobs = compute_logprobs_batch(...)
model.load_state_dict(cur_state)         # back to training model
```

### 3. Separate Inference Model with UVM/Offloading

For even larger models, Megatron supports:
- A **separate inference model** with different parallelism than the training model
- **Unified Virtual Memory (UVM)** to seamlessly page inference model weights between CPU and GPU
- **`torch_memory_saver`** for explicit CPU/GPU weight swapping

### 4. CUDA Graph Caching

Inference can use CUDA graphs for speed:
```python
with megatron_rl_inference_mode(model, optimizer, cuda_graph_impl='local', ...):
    # Inference runs with CUDA graph capturing
    # CUDA graphs eliminate kernel launch overhead
```

During training, CUDA graphs are toggled off to allow dynamic shapes.

---

## 9.8 Multi-Turn and Multi-Task RL

### Multi-Turn Rollouts

Megatron-RL supports **multi-turn conversations** as rollouts:

```python
class TokenRollout:
    trajectory: list[list[int]]  # list of TURNS, not a single sequence
    # e.g., [[user_tokens], [assistant_tokens], [user_tokens], [assistant_tokens]]
```

Each turn in a multi-turn trajectory:
- Gets its own trajectory segment
- Shares the same reward (the final reward for the whole conversation)
- Is treated as a separate training sample during the update

### Multi-Task Training

The `WeightedMultiTask` agent manages multiple environments:

```yaml
# Example YAML config for multi-task RL
- agent_type: "megatron.rl.agent.math_agent.MathAgent"
  agent_args:
    dataset: "gsm8k"
  weight: 0.5

- agent_type: "megatron.rl.agent.code_agent.CodeAgent"
  agent_args:
    dataset: "humaneval"
  weight: 0.3

- agent_type: "megatron.rl.agent.eval_agent.EvalAgent"
  agent_args:
    dataset: "mmlu"
  weight: 0.0
  evaluation_only: true
```

Rollouts are sampled proportionally to weights across environments.

---

## 9.9 Staleness Tracking

In large-scale RL, rollouts may be **stale** — generated by an older version of the policy:

```python
class RolloutStats:
    policy_staleness: list[list[int]]    # How many updates since this rollout
    kv_cache_staleness: list[list[int]]  # How stale the KV cache was
    completed_at_steps: list[list[int]]  # When each turn completed
```

Megatron tracks staleness per-token and per-turn, and reports these as training metrics. High staleness can degrade training quality because the importance sampling correction becomes less accurate.

The **importance sampling truncation coefficient** (`--rl-importance-sampling-truncation-coef`) caps the IS weight to prevent extreme corrections:

```python
is_weights = (old_logprobs - inference_logprobs).exp()
is_weights = torch.min(is_weights, truncation_coef)  # Cap the correction
```

---

## 9.10 Key Files Covered

| File | Purpose |
|------|---------|
| `train_rl.py` | Entry point, `forward_step`, `loss_func`, dataset provider |
| `megatron/rl/rl_utils.py` | Core GRPO logic: `calculate_grpo_loss`, `prepare_data_for_update`, `get_environment_rollouts`, `get_logprobs`, rollout collection |
| `megatron/rl/agent/api.py` | Agent abstractions: `Rollout`, `TokenRollout`, `GroupedRolloutGenerator`, `EvaluationAgent` |
| `megatron/rl/agent/weighted_multi_task.py` | Multi-task agent with weighted environment sampling |
| `megatron/rl/inference/inference_interface.py` | `InferenceInterface` base class for generation |
| `megatron/rl/inference/megatron.py` | `MegatronLocal` inference backend |
| `megatron/rl/parallel_utils.py` | Process group construction for inference models |
| `megatron/rl/sequence_packing_utils.py` | Sequence packing for RL: bin-packing, `cu_seqlens`, microbatch creation |
| `megatron/rl/logging.py` | RL-specific logging and metrics |
| `megatron/rl/README.md` | Design documentation for Megatron-RL |

---

## 9.11 Interview-Ready Takeaways

### GRPO vs PPO — Why GRPO Wins for Reasoning

| | PPO | GRPO |
|--|-----|------|
| Models needed | 4 (policy, ref, reward, value) | 2 (policy, ref) |
| Reward signal | Learned reward model | Verifiable rewards (math, code) |
| Critic | Value model estimates V(s) | Group statistics replace critic |
| Memory | ~4× model size | ~2× model size |
| Best for | General preference alignment | Verifiable reasoning tasks |

### The GRPO Advantage Formula

> "For each prompt, sample K responses. Normalize rewards within the group. Good responses get positive advantage, bad ones get negative. No critic needed."

```
A(yᵢ) = (rᵢ - μ_group) / (σ_group + ε)
```

### The Three Logprob Computations

Every GRPO step requires three forward passes:
1. **π_θ** (current policy, **with gradient**) — the training forward pass
2. **π_old** (the policy at rollout time, no grad) — for the probability ratio
3. **π_ref** (the frozen reference policy, no grad) — for the KL penalty

Megatron optimizes this by weight-swapping instead of keeping separate models.

### Why Clipping Matters

> "Without clipping, a single very-good response could cause a huge policy update, destabilizing training. The clip at 1±ε ensures each update is small."

### Memory Management Interview Answer

> "RL training alternates between inference (generation) and training (gradient updates). You only need optimizer states during training, so Megatron offloads them to CPU during inference. Similarly, it swaps reference model weights in/out rather than keeping a permanent copy."

### Policy Staleness

> "In large-scale RL, rollouts are collected while the model is still training. By the time you use a rollout for training, the policy may have changed. Megatron tracks staleness and applies importance sampling correction: weight each sample by π_old/π_inference to account for the distribution shift."

### Sequence Packing in RL

> "Same idea as SFT packing: multiple variable-length rollouts are packed into fixed-size bins using cu_seqlens for FlashAttention boundaries. This avoids wasting compute on padding tokens, which is especially important in RL where response lengths vary wildly."
