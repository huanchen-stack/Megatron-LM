# Chapter 2: Data Loading & Dataset Construction

## From Binary Files to Training Batches

In Chapter 1, we preprocessed raw text into binary `.bin/.idx` files. Now the question becomes: how do those files turn into the `tokens`, `labels`, `loss_mask`, `attention_mask`, and `position_ids` tensors that the model actually consumes?

The answer involves a sophisticated multi-layer data pipeline:

```
On disk:  .bin/.idx files (memory-mapped binary)
              │
              ▼
Layer 1:  IndexedDataset          ← O(1) random access to documents
              │
              ▼
Layer 2:  GPTDataset              ← Constructs fixed-length training samples
              │                      from variable-length documents
              ▼
Layer 3:  BlendedDataset          ← Mixes multiple GPTDatasets with weights
              │
              ▼
Layer 4:  DataLoader + Sampler    ← Distributes samples across DP ranks
              │
              ▼
Output:   {tokens, labels, loss_mask, attention_mask, position_ids}
```

## Layer 1: IndexedDataset — Memory-Mapped Access

We covered the format in Chapter 1. What matters here is the access pattern:

```python
# In megatron/core/datasets/indexed_dataset.py
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, path_prefix, ...):
        # Memory-map the binary — no data loaded until accessed
        self.bin_buffer_mmap = numpy.memmap(bin_path, mode='r', order='C')
        self.bin_buffer = memoryview(self.bin_buffer_mmap)
        # Load index metadata (sequence lengths, byte pointers, document boundaries)
        self._index = _IndexReader(idx_path, ...)
```

Getting document `i` is a pointer lookup:
```python
def get(self, idx, offset=0, length=None):
    # Look up byte pointer and read from memory-mapped buffer
    ptr, size = self._index[idx]  # O(1) — just array indexing
    # Read directly from memory map — zero-copy
    return numpy.frombuffer(self.bin_buffer, dtype=self._index.dtype,
                           count=size, offset=ptr)
```

**Key insight**: The `IndexedDataset` stores **documents** (variable-length sequences of token IDs). A document might be 50 tokens (a tweet) or 50,000 tokens (an article). The `GPTDataset` layer above is responsible for stitching these into fixed-length training sequences.

## Layer 2: GPTDataset — Building Fixed-Length Samples

The `GPTDataset` (in `megatron/core/datasets/gpt_dataset.py`, ~900 lines) is the heart of the data pipeline. Its job: take a collection of variable-length documents and produce N fixed-length training samples of exactly `sequence_length` tokens.

### The Three-Index Trick

Building samples from documents isn't trivial. Documents have different lengths, and training needs fixed-length sequences. GPTDataset creates three indices at initialization:

**1. Document Index (`document_index`)** — a shuffled list of document IDs:
```
Given 5 documents [d0, d1, d2, d3, d4] and we need 3 epochs:
document_index = [d3, d1, d0, d4, d2, d1, d3, d0, d4, d2, d0, d1, d2, d4, d3]
                  ├─── epoch 1 ────┤├─── epoch 2 ────────┤├─── epoch 3 ────────┤
```
Documents are shuffled per epoch using the random seed.

**2. Sample Index (`sample_index`)** — maps from sample index to (document_index position, offset):
```
Given sequence_length = 1024:
sample_index[0] = (0, 0)       # Start at doc_index[0], offset 0
sample_index[1] = (0, 1024)    # doc_index[0] is long — still in same doc
sample_index[2] = (1, 512)     # doc_index[0] ended, continued into doc_index[1]
sample_index[3] = (3, 0)       # doc_index[1:3] were short, skipped ahead
```
This handles the key challenge: **documents don't align with sequence boundaries**. A sample might span multiple short documents, or a long document might yield multiple samples.

**3. Shuffle Index (`shuffle_index`)** — final shuffling of samples:
```
shuffle_index = [4, 0, 2, 6, 1, 9, 5, 8, 7, 3]
```

### How `__getitem__` Works

When the DataLoader asks for sample `k`:

```python
def __getitem__(self, idx):
    text, _ = self._query_document_sample_shuffle_indices(idx)
    text = torch.from_numpy(text).long()

    # Split into input tokens and target labels (shifted by 1)
    tokens = text[:-1].contiguous()    # Input: [t0, t1, ..., t_{n-1}]
    labels = text[1:].contiguous()     # Target: [t1, t2, ..., t_n]

    # Create masks and position IDs
    attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        tokens, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss
    )

    return {"tokens": tokens, "labels": labels, "attention_mask": attention_mask,
            "loss_mask": loss_mask, "position_ids": position_ids}
```

The internal `_query_document_sample_shuffle_indices` method chains through all three indices:
1. `shuffle_index[k]` → get the real sample index `j`
2. `sample_index[j]` and `sample_index[j+1]` → get start/end positions in document index
3. Walk through `document_index` entries and read token spans from `IndexedDataset`
4. Concatenate spans to form one `sequence_length + 1` token sequence

### What Each Output Tensor Means

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `tokens` | `[seq_len]` | Input token IDs fed to the embedding layer |
| `labels` | `[seq_len]` | Target token IDs — `tokens` shifted right by 1 |
| `loss_mask` | `[seq_len]` | 1.0 for real tokens, 0.0 for padding. Optionally 0.0 for EOD tokens (`--eod-mask-loss`) |
| `attention_mask` | `[1, seq_len, seq_len]` | Causal mask. Optionally resets at document boundaries (`--reset-attention-mask`) |
| `position_ids` | `[seq_len]` | Usually `[0, 1, 2, ...]`. Optionally resets at document boundaries (`--reset-position-ids`) |

### Caching for Performance

Index construction (building document_index, sample_index, shuffle_index) is expensive — it involves shuffling and walking through all documents. So Megatron:
1. **Builds on rank 0** only
2. **Saves the indices as `.npy` files** to the `--data-cache-path`
3. **Other ranks load from cache** via `numpy.load(..., mmap_mode='r')`
4. Cache keys include a hash of the dataset path, seed, sequence length, etc. — so changing any config produces a new cache

## Layer 3: BlendedDataset — Mixing Data Sources

Real training uses multiple data sources with different proportions. For example, you might train on:
- 70% web text
- 20% code
- 10% math

The `BlendedDataset` (in `megatron/core/datasets/blended_dataset.py`) wraps multiple `GPTDataset` instances and samples from them according to weights:

```python
class BlendedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, weights, size, config):
        # datasets: [GPTDataset_web, GPTDataset_code, GPTDataset_math]
        # weights:  [0.7, 0.3, 0.1] (normalized automatically)
        self.dataset_index, self.dataset_sample_index = self._build_indices()

    def __getitem__(self, idx):
        dataset_idx = self.dataset_index[idx]       # Which dataset?
        sample_idx = self.dataset_sample_index[idx]  # Which sample in that dataset?
        return self.datasets[dataset_idx][sample_idx]
```

The blending algorithm ensures samples are drawn proportionally without replacement until all datasets contribute their share. At each step, it draws from whichever dataset has the largest gap between target and actual proportion.

### How Blending Is Configured

At training time, you specify blending on the command line:
```bash
--data-path 0.7 /preprocessed/web_text 0.2 /preprocessed/code 0.1 /preprocessed/math
```

Or per-split:
```bash
--train-data-path 0.7 /data/web 0.3 /data/code
--valid-data-path 1.0 /data/valid
--test-data-path 1.0 /data/test
```

The `BlendedMegatronDatasetBuilder` creates the full pipeline: `IndexedDataset` → `GPTDataset` → `BlendedDataset` for each split (train/valid/test).

## Layer 4: Data Sampling Across Ranks

The `MegatronPretrainingSampler` (in `megatron/training/datasets/data_samplers.py`) ensures each data-parallel rank sees different samples:

```python
class MegatronPretrainingSampler:
    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, ...):
        self.last_batch_size = total_samples % (micro_batch_size * data_parallel_size)
```

Given 8 data-parallel ranks and batch size 4, sample distribution looks like:
```
Rank 0: samples [0, 1, 2, 3], [32, 33, 34, 35], ...
Rank 1: samples [4, 5, 6, 7], [36, 37, 38, 39], ...
...
Rank 7: samples [28, 29, 30, 31], [60, 61, 62, 63], ...
```

Each rank draws from a contiguous, non-overlapping slice of the total dataset.

## SFT Dataset — Fine-Tuning with Chat Data

For supervised fine-tuning (SFT), the data format changes. Instead of raw text, you have conversations:

```json
{"messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"}
]}
```

The `SFTDataset` (in `megatron/training/datasets/sft_dataset.py`) handles this:

```python
class SFTDataset(MegatronDataset):
    def __getitem__(self, idx):
        merged_conversations = self.dataset[self.indices[idx % len(self.indices)]]
        split_conversations = self._split_conversations(merged_conversations)

        pack_tokens, pack_targets = [], []
        for conversation in split_conversations:
            tokens, targets = tokenizer.tokenize_conversation(
                conversation, return_target=True
            )
            pack_tokens.extend(tokens.tolist())
            pack_targets.extend(targets.tolist())
```

**Critical difference from pretraining**: During SFT, the `loss_mask` is set to `IGNORE_INDEX = -100` for all tokens that are NOT assistant responses. This means the model only learns to generate assistant text, not to reproduce system prompts or user questions.

SFT also supports **packing multiple conversations** into a single sequence for efficiency. The `cu_seqlens` (cumulative sequence lengths) array tracks where each conversation starts and ends within the packed sequence, so attention doesn't leak across conversations.

## FIM Dataset — Fill-in-the-Middle for Code Models

The `GPTFIMDataset` (in `megatron/training/datasets/fim_dataset.py`) extends `GPTDataset` with Fill-in-the-Middle augmentation, used when training code models:

Given a code sample: `prefix | middle | suffix`

**PSM (Prefix-Suffix-Middle) format:**
```
[FIM_PREFIX] prefix [FIM_SUFFIX] suffix [FIM_MIDDLE] middle
```

**SPM (Suffix-Prefix-Middle) format:**
```
[FIM_PREFIX] [FIM_SUFFIX] suffix [FIM_MIDDLE] prefix middle
```

FIM is applied probabilistically (controlled by `--fim-rate`) so the model learns both standard left-to-right and fill-in-the-middle generation.

## The Full Data Flow: Putting It All Together

Here's what happens when `pretrain_gpt.py` starts training:

```python
# In pretrain_gpt.py
def train_valid_test_datasets_provider(train_val_test_num_samples):
    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset      # Chat format
    elif args.mock_data:
        dataset_type = MockGPTDataset  # Random data for testing
    elif args.fim_data:
        dataset_type = GPTFIMDataset   # Code with FIM augmentation
    else:
        dataset_type = GPTDataset      # Standard pretraining

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()
    return train_ds, valid_ds, test_ds
```

The builder:
1. Opens each `.bin/.idx` pair as an `IndexedDataset`
2. Wraps each in a `GPTDataset` (or `SFTDataset`, etc.) with the right split indices
3. If multiple data paths are given, wraps them all in a `BlendedDataset` with the specified weights
4. Returns train, valid, test datasets

## Key Files Covered

| File | Purpose |
|------|---------|
| `megatron/core/datasets/indexed_dataset.py` | Memory-mapped binary dataset access |
| `megatron/core/datasets/gpt_dataset.py` | `GPTDataset` — builds fixed-length samples from variable-length documents |
| `megatron/core/datasets/megatron_dataset.py` | `MegatronDataset` — abstract base class with caching logic |
| `megatron/core/datasets/blended_dataset.py` | `BlendedDataset` — mixes multiple datasets with weights |
| `megatron/core/datasets/blended_megatron_dataset_builder.py` | Builder that constructs the full dataset pipeline |
| `megatron/core/datasets/blended_megatron_dataset_config.py` | Configuration dataclass |
| `megatron/core/datasets/data_schedule.py` | Data scheduling (ordering) |
| `megatron/training/datasets/sft_dataset.py` | `SFTDataset` — chat-format data with loss masking |
| `megatron/training/datasets/fim_dataset.py` | `GPTFIMDataset` — fill-in-the-middle for code |
| `megatron/training/datasets/data_samplers.py` | `MegatronPretrainingSampler` — distributes samples across ranks |

## Interview-Ready Takeaways

1. **Three-index mapping**: GPTDataset uses document_index → sample_index → shuffle_index to efficiently construct fixed-length samples from variable-length documents. This is the core intellectual contribution of the data pipeline.

2. **Samples can span documents**: A single training sample may contain the end of one document, an EOD token, and the beginning of the next. This is by design — it maximizes GPU utilization by avoiding padding.

3. **Dataset blending** lets you train on a weighted mixture of data sources (web text, code, math) without preprocessing them together. Weights are specified on the command line and sampling is done at training time.

4. **SFT loss masking** is the key difference between pretraining and fine-tuning data: during SFT, loss is computed only on assistant responses, not on system/user tokens. This is implemented via the `IGNORE_INDEX = -100` convention.

5. **Indices are cached and distributed**: Rank 0 builds the indices and saves them as `.npy` files. Other ranks load them via memory-mapped reads. This avoids redundant computation across thousands of ranks.

6. **Fill-in-the-Middle (FIM)** is a data augmentation for code models that rearranges prefix/middle/suffix so the model learns to infill — it's applied probabilistically during data loading, not during preprocessing.

7. **Everything is deterministic**: Given the same random seed, sequence length, and dataset, every rank will produce the exact same samples in the exact same order. This is critical for reproducibility and debugging at scale.

---

*Next up: [Chapter 3 — Model Architecture Deep Dive](./ch03_model_architecture.md) explores how GPTModel is constructed from TransformerBlocks, attention layers, and MLPs.*
