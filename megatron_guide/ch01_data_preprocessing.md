# Chapter 1: Data Preprocessing Pipeline

## Why You Can't Just Read Text Files

Here's the problem: you have 1,024 GPUs ready to train a 70B-parameter model. Each GPU needs a fresh batch of tokenized text every ~100 milliseconds. If your training data lives in JSON files, every batch requires opening a file, parsing JSON, tokenizing text, and converting to tensors — all at runtime. At 1,024 GPUs, that's 10,000+ file opens per second. Your filesystem will melt.

Megatron's solution: **preprocess once, train many times**. Convert your raw text corpus into a binary format that supports memory-mapped, O(1) random access. During training, reading a sample is just a pointer dereference — no parsing, no deserialization, no filesystem pressure.

The pipeline looks like this:

```
Raw Text (.jsonl)
    │
    ▼
┌─────────────────────┐
│  preprocess_data.py  │  ← tokenize + write binary
│  (multiprocessing)   │
└─────────────────────┘
    │
    ▼
Binary Files (.bin + .idx)
    │
    ▼
Training (memory-mapped, O(1) access)
```

## The Entry Point: `tools/preprocess_data.py`

This is the script you actually run. A typical invocation:

```bash
python tools/preprocess_data.py \
    --input my_corpus.jsonl \
    --output-prefix my-gpt2 \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --workers 32
```

The script is about 400 lines and does three things:
1. Reads JSON lines from the input file
2. Tokenizes each document using a multiprocessing pool
3. Writes the resulting token IDs into the IndexedDataset binary format

### Key Arguments

| Argument | What It Does |
|----------|-------------|
| `--input` | Path to `.jsonl` file. Each line must have a JSON object with the key(s) you want to tokenize |
| `--json-keys` | Which JSON keys to extract (default: `["text"]`). Each key produces its own `.bin/.idx` pair |
| `--output-prefix` | Output path prefix. Produces `{prefix}_{key}_document.bin` and `.idx` |
| `--tokenizer-type` | Which tokenizer to use: `GPT2BPETokenizer`, `SentencePieceTokenizer`, `HuggingFaceTokenizer`, etc. |
| `--append-eod` | Append an end-of-document token after each document (almost always want this) |
| `--workers` | Number of parallel tokenization workers |
| `--partitions` | Split input across N partitions for maximum parallelism |
| `--split-sentences` | Use NLTK to split documents into sentences before tokenizing |

### The Encoder Class

The core logic lives in the `Encoder` class. Here's the actual code:

```python
class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Each worker process calls this to create its own tokenizer
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            # ... NLTK sentence splitter setup ...
        else:
            Encoder.splitter = IdentitySplitter()

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            sentences = [text] if not isinstance(text, list) else text
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)
```

Key details:
- **Each worker has its own tokenizer instance** (created in `initializer()`), so tokenization is embarrassingly parallel
- **`encode()` returns both token IDs and sentence lengths** — sentence boundaries are preserved in the index for downstream use
- **The EOD token is appended to each document** — this is critical because during training, GPTDataset concatenates documents and needs to know where one ends and another begins

### The Processing Pipeline

The `Partition` class orchestrates the processing:

```python
def process_json_file(self, file_name):
    input_file_name, output_prefix = file_name
    fin = open(input_file_name, 'r', encoding='utf-8')
    
    encoder = Encoder(self.args)
    tokenizer = build_tokenizer(self.args)
    pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 32)

    builders = {}
    for key in self.args.json_keys:
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

    for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
        for key in doc.keys():
            builders[key].add_document(doc[key], sentence_lens[key])

    builders[key].finalize(output_idx_files[key])
```

The flow:
1. Open the input JSONL file
2. Create a multiprocessing pool where each worker has a tokenizer
3. Stream documents through `pool.imap()` — this maps `encode()` over every line, 32 at a time
4. For each encoded document, call `builder.add_document()` to write it to the binary format
5. Call `builder.finalize()` to write the index file

**Multi-partition mode**: When `--partitions > 1`, the script splits the input into N chunks, processes them in parallel as separate processes, then merges the resulting `.bin/.idx` files using `IndexedDatasetBuilder.add_index()`.

## Tokenizer Architecture

When `build_tokenizer(args)` is called, it creates the right tokenizer based on `--tokenizer-type`. The tokenizer code lives in `megatron/core/tokenizers/`.

```
megatron/core/tokenizers/
├── __init__.py             # MegatronTokenizer base class + from_pretrained()
├── base_tokenizer.py       # AbstractTokenizer and BaseTokenizer ABCs
├── megatron_tokenizer.py   # Full MegatronTokenizer with encode/decode/special tokens
├── text/                   # Text tokenizer implementations
│   ├── gpt2_tokenizer.py   # GPT-2 BPE tokenizer
│   └── sentencepiece_tokenizer.py  # SentencePiece tokenizer (used for LLaMA etc.)
├── utils/
│   └── build_tokenizer.py  # The factory function
└── vision/                 # Vision tokenizers (for multimodal)
```

### The Factory: `build_tokenizer()`

Located at `megatron/core/tokenizers/utils/build_tokenizer.py`, this function maps `--tokenizer-type` to the right class:

| `--tokenizer-type` | Class | Use Case |
|---------------------|-------|----------|
| `GPT2BPETokenizer` | `GPT2Tokenizer` | GPT-2 style models (needs `--vocab-file` and `--merge-file`) |
| `SentencePieceTokenizer` | `SentencePieceTokenizer` | LLaMA, Mistral, etc. (needs `--tokenizer-model`) |
| `HuggingFaceTokenizer` | HuggingFace `AutoTokenizer` | Any HuggingFace model (needs `--tokenizer-model` as HF model name) |
| `NullTokenizer` | `NullTokenizer` | Testing/debugging only |

All tokenizers share the same interface:
- `.tokenize(text) → List[int]` — convert text to token IDs
- `.detokenize(ids) → str` — convert token IDs back to text  
- `.eod` — the end-of-document token ID
- `.vocab_size` — vocabulary size

## The IndexedDataset Binary Format

This is the format that makes Megatron training fast. It's defined in `megatron/core/datasets/indexed_dataset.py` (~1000 lines) and consists of two files:

### The `.bin` File (Data)
Raw token IDs stored as a flat numpy array. If your vocabulary is < 65,500 tokens, IDs are stored as `uint16` (2 bytes each). Otherwise, `int32` (4 bytes each). That's it — no headers, no separators, just a continuous stream of numbers.

```
.bin file (conceptual):
[tok_0, tok_1, ..., tok_N, tok_0, tok_1, ..., tok_M, ...]
 ├── Document 0 ──────────┤├── Document 1 ──────────┤
```

### The `.idx` File (Index)
Metadata that tells you where each document starts and ends:

```
.idx file layout:
┌──────────────────────┐
│ Header (magic bytes)  │  "MMIDIDX\x00\x00"
│ Version               │  
│ DType code            │  Which numpy dtype was used in .bin
│ Num sequences         │  Total number of sequences
│ Num documents         │  Total number of documents
├──────────────────────┤
│ Sequence lengths      │  Array: length of each sequence
│ Sequence pointers     │  Array: byte offset of each sequence in .bin
│ Document indices      │  Array: which sequences belong to which document
│ Sequence modes        │  Array: mode per sequence (for multimodal)
└──────────────────────┘
```

### Why This Format?

The key insight is **memory-mapped I/O**. When you open an IndexedDataset:

```python
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, path_prefix, ...):
        # Memory-map the .bin file — no data is loaded into RAM
        self.bin_buffer_mmap = numpy.memmap(bin_path, mode='r', order='C')
        self.bin_buffer = memoryview(self.bin_buffer_mmap)
```

- The `.bin` file is **memory-mapped** via `numpy.memmap()` — the OS maps the file into virtual memory without loading it. Pages are loaded on-demand when accessed.
- Reading document `i` is a pointer arithmetic operation: look up the byte offset in the `.idx`, read that many bytes from the `.bin`.
- **Zero deserialization**: token IDs are already in the right numpy dtype. Just cast the memory buffer.
- **O(1) random access**: any document can be accessed in constant time, regardless of dataset size.
- **No RAM pressure**: a 1TB dataset on disk uses essentially zero RAM until you access specific pages.

### IndexedDatasetBuilder

The builder writes the binary format:

```python
builders[key] = indexed_dataset.IndexedDatasetBuilder(
    output_bin_files[key],
    dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
)

# For each document:
builders[key].add_document(doc_ids, sentence_lens)

# When done:
builders[key].finalize(output_idx_files[key])
```

`add_document()` appends token IDs to the `.bin` file and records sequence lengths and byte offsets. `finalize()` writes the `.idx` file with all the accumulated metadata.

## Other Preprocessing Tools

### `tools/merge_datasets.py`
Merges multiple `.bin/.idx` pairs into one. Useful when you've preprocessed data in parallel or from multiple sources:
```bash
python tools/merge_datasets.py \
    --input part_0 part_1 part_2 \
    --output-prefix merged
```

### `tools/preprocess_mmdata.py`
Variant for multimodal data. Instead of just text, it handles interleaved text and images, writing sequence "modes" (text vs. image) into the `.idx` file.

### `tools/build_sequences_per_dataset.py`
For large-scale runs with hundreds of data files, this script pre-computes metadata (sequence counts per file) into a single JSON, so the dataloader doesn't need to open every `.idx` file at startup.

## Practical Walkthrough

If you have a 100GB text corpus in JSONL format and want to preprocess it for GPT training:

```bash
# Step 1: Preprocess with 64 workers, 8 partitions for parallelism
python tools/preprocess_data.py \
    --input /data/my_corpus.jsonl \
    --output-prefix /preprocessed/my_corpus \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --append-eod \
    --workers 64 \
    --partitions 8

# This produces:
#   /preprocessed/my_corpus_text_document.bin  (binary token IDs)
#   /preprocessed/my_corpus_text_document.idx  (index file)

# Step 2: To train with multiple data sources, blend them:
# (blending happens at training time, not preprocessing time)
# --data-path weight1 path1 weight2 path2 ...
torchrun ... pretrain_gpt.py \
    --data-path 0.7 /preprocessed/web_text 0.3 /preprocessed/code_text \
    ...
```

## Key Files Covered

| File | Purpose |
|------|---------|
| `tools/preprocess_data.py` | Main preprocessing script — tokenizes JSONL and writes binary |
| `megatron/core/tokenizers/utils/build_tokenizer.py` | Factory function creating the right tokenizer |
| `megatron/core/tokenizers/text/gpt2_tokenizer.py` | GPT-2 BPE tokenizer |
| `megatron/core/tokenizers/text/sentencepiece_tokenizer.py` | SentencePiece tokenizer (LLaMA, etc.) |
| `megatron/core/datasets/indexed_dataset.py` | Binary dataset format: `IndexedDataset` and `IndexedDatasetBuilder` |
| `tools/merge_datasets.py` | Merge multiple `.bin/.idx` pairs |
| `tools/preprocess_mmdata.py` | Multimodal data preprocessing |
| `tools/build_sequences_per_dataset.py` | Pre-compute per-dataset metadata for fast init |

## Interview-Ready Takeaways

1. **Preprocessing is a one-time cost**: You tokenize your corpus once and write it to a binary format. Training reads this binary directly — no runtime tokenization.

2. **The IndexedDataset format** uses two files: `.bin` (flat array of token IDs) and `.idx` (byte offsets and document boundaries). This enables O(1) random access via memory-mapped I/O.

3. **Memory-mapped I/O** (`numpy.memmap`) maps the binary file into virtual memory without loading it. The OS loads pages on-demand, so a 1TB dataset uses near-zero RAM.

4. **Preprocessing is embarrassingly parallel**: The `Encoder` class runs in a multiprocessing pool, and the `--partitions` flag enables multi-process pipeline parallelism across file chunks.

5. **The dtype is chosen automatically**: If vocab size < 65,500, token IDs are stored as `uint16` (2 bytes). Otherwise `int32` (4 bytes). This halves storage for most modern tokenizers.

6. **EOD tokens are critical**: Each document ends with an EOD token. During training, documents are concatenated into sequences, and the EOD token tells the model where one document ends and another begins.

7. **The tokenizer is pluggable**: Megatron supports GPT-2 BPE, SentencePiece, and HuggingFace tokenizers through the same `build_tokenizer()` factory. For modern models (LLaMA, Qwen, Mistral), you'd use `HuggingFaceTokenizer`.

---

*Next up: [Chapter 2 — Data Loading & Dataset Construction](./ch02_data_loading.md) shows how these binary files get loaded, blended, and converted into training batches during training.*
