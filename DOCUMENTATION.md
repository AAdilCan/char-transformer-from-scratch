# Documentation

Technical deep-dive for `char-transformer-from-scratch`: what it is, how it's
built, how it was trained, and the engineering decisions behind it.

## 1. Overview

This is a decoder-only Transformer language model — the same family as GPT —
built from first principles in PyTorch and trained at the character level on the
Tiny Shakespeare corpus. "From scratch" is the point: I deliberately did **not**
use `torch.nn.Transformer`, `nn.MultiheadAttention`, or any pretrained
component. The Q/K/V projections, head splitting, scaled dot-product attention,
causal masking, residual blocks, weight initialization, learning-rate schedule,
training loop, and autoregressive sampler are all written by hand so every
moving part is inspectable.

The goal was understanding, not state-of-the-art results: I wanted to be able to
point at each line and say what it does and why. The deliverable is a small
(0.83M-parameter) model that trains on a laptop CPU in a single run and produces
recognizably Shakespeare-shaped text.

## 2. Architecture

### Module structure

```
                          scripts/train.py
                                 │
              ┌──────────────────┼───────────────────┐
              ▼                  ▼                    ▼
        data.build_dataset   GPT (model.py)      Trainer (trainer.py)
              │                  │                    │
       ┌──────┴──────┐    ┌──────┴───────┐     ┌──────┴───────────┐
       ▼             ▼    ▼              ▼     ▼                  ▼
 CharTokenizer  get_batch  Block × N   generate  cosine_lr   save_checkpoint
 (tokenizer.py) (data.py)  (attn+ffwd)          (lr_schedule) (checkpoint.py)

        scripts/generate.py ──► load_checkpoint ──► GPT.generate
        scripts/evaluate.py ──► load_checkpoint ──► evaluate + plotting
```

Each module has a single responsibility:

| Module | Responsibility |
|--------|----------------|
| `config.py` | Typed dataclasses (`ModelConfig`, `TrainConfig`, `DataConfig`, `Config`) with YAML load and `to_dict` for serialization. Validates `n_embd % n_head == 0`. |
| `tokenizer.py` | `CharTokenizer`: deterministic char↔id mapping built from the sorted unique characters of the corpus; JSON save/load. |
| `data.py` | Corpus loading, contiguous train/val split, and `get_batch` random-window sampling. |
| `model.py` | The model: `CausalSelfAttention`, `FeedForward`, `Block`, and the top-level `GPT` with `forward` (returns logits + optional loss) and `generate`. |
| `lr_schedule.py` | `cosine_lr`: a pure `step → lr` function (warmup + cosine decay), independently testable. |
| `trainer.py` | `Trainer` orchestration: AdamW with grouped weight decay, LR scheduling, periodic loss estimation, best-by-val checkpointing. |
| `evaluation.py` | `evaluate` / `perplexity`: turns mean cross-entropy into perplexity using the same estimator as training. |
| `plotting.py` | Renders loss curves and the LR schedule to PNG (headless `Agg` backend). |
| `checkpoint.py` | Bundles weights + config + tokenizer + optimizer + history into one `.pt`, and rebuilds the model from it. |
| `utils.py` | Device resolution and RNG seeding. |

### The forward pass

For an input batch of token ids `idx` with shape `(B, T)`:

1. **Embed.** `token_embedding(idx) + position_embedding(arange(T))` → `(B, T, C)`. Positions are *learned*, not sinusoidal.
2. **Blocks.** `N` pre-norm blocks. Each does `x = x + attn(ln1(x))` then `x = x + ffwd(ln2(x))`.
3. **Attention.** A single `Linear(C, 3C)` produces Q, K, V; each is reshaped to `(B, n_head, T, head_dim)`. Scores are `Q·Kᵀ / sqrt(head_dim)`, masked with a registered lower-triangular buffer so position `t` only sees `≤ t`, softmaxed, dropped out, and applied to V. Heads are concatenated and projected back to `C`.
4. **Head.** Final LayerNorm, then `Linear(C, vocab_size)` → logits `(B, T, vocab_size)`.
5. **Loss.** If targets are supplied, mean cross-entropy over all `B·T` positions.

See [`model.py:CausalSelfAttention.forward`](src/char_transformer/model.py) for
the hand-rolled attention.

## 3. Data

- **Source:** Tiny Shakespeare (`input.txt`), the concatenated works used by
  Karpathy's char-RNN/minGPT work. Downloaded by
  [`scripts/download_data.py`](scripts/download_data.py).
- **Size:** 1,115,394 characters, 40,000 lines, **65 unique characters**
  (letters in both cases, digits, punctuation, newline, space).
- **Schema:** raw UTF-8 text. The "schema" is implicit: lines of dialogue with
  ALL-CAPS speaker labels followed by a colon and a newline.
- **Tokenization:** character level. The vocabulary is the sorted set of unique
  characters, so `vocab_size = 65` and each character is one integer id. Sorting
  makes the mapping deterministic across machines, which keeps a serialized
  vocabulary stable.
- **Split:** a contiguous 90/10 train/val split — the validation set is the
  *last* 10% of the text, held out as a block the model never trains on. This
  mirrors the real task (continue unseen prose) better than shuffling windows,
  which would leak nearly-identical neighboring windows across the boundary.
- **Batching:** `get_batch` draws random start offsets and slices
  `block_size`-length windows; the target is the input shifted right by one, so
  every position is a next-character prediction. With a 1 MB corpus there is no
  need for a `DataLoader` or epochs — random crops give effectively unlimited
  fresh batches.

## 4. Methodology

### Model

A 4-layer, 4-head, 128-dim decoder-only Transformer (`block_size = 128`,
dropout 0.1), **826,368 parameters**. Design choices:

- **Pre-norm blocks** (LayerNorm before each sublayer, residual added after)
  rather than the original post-norm. Pre-norm keeps a clean identity path
  through the residual stream and trains stably on CPU without architecture-
  specific LR tuning.
- **GPT-2 weight init:** normal(0, 0.02) for linears and embeddings, and
  residual-projection weights additionally scaled by `1/sqrt(2·n_layer)` to keep
  activation variance from growing with depth.
- **GELU 4× feed-forward**, the conventional expansion ratio.
- **Learned positional embeddings** over sinusoidal: simpler, and the fixed
  `block_size` means there's no need to extrapolate to unseen lengths.

### Optimization

- **AdamW** (`betas = (0.9, 0.99)`) with **decoupled weight decay applied only
  to 2-D weight matrices** — biases, embeddings, and LayerNorm gains are left
  undecayed. This is the standard GPT setup; regularizing 1-D parameters toward
  zero hurts more than it helps.
- **LR schedule:** linear **warmup over 200 steps** to a peak of `3e-4`, then
  **cosine decay** to a floor of `0.1 × peak`. Warmup keeps the first updates
  small while Adam's moment estimates are still noisy; cosine decay is a smooth,
  well-tested anneal. See [`lr_schedule.py`](src/char_transformer/lr_schedule.py).
- **Gradient clipping** at global norm 1.0.
- **Loss estimation:** train/val loss is averaged over 200 random batches with
  dropout off every 250 steps. A single batch is far too noisy to compare across
  eval points; averaging a handful gives a curve worth plotting and a stable
  signal for best-checkpoint selection.

### Generation

Autoregressive sampling ([`model.py:GPT.generate`](src/char_transformer/model.py)):
at each step the context is cropped to the last `block_size` characters, logits
for the final position are divided by **temperature**, optionally restricted to
the **top-k** highest logits, softmaxed, and sampled from. Lower temperature is
greedier; top-k removes the long tail of unlikely characters that otherwise
accumulate into nonsense over a long sample.

### Alternatives considered

- **Subword/BPE tokenization** — rejected: it would mean training a tokenizer
  and a much larger embedding/output matrix, defeating the CPU-trainable goal.
  Character level keeps the vocabulary at 65 and the whole forward pass cheap.
- **`nn.MultiheadAttention` / `nn.Transformer`** — rejected on purpose; the
  entire point is to write attention by hand.
- **Flash / fused attention** — unnecessary at `T = 128` on CPU and would hide
  the masked-softmax logic.

## 5. Results

Single run, 5,000 steps, seed 1337, CPU. Metrics re-estimated over 200 batches
([`figures/metrics.json`](figures/metrics.json)):

| Metric                | Train | Validation |
|-----------------------|-------|------------|
| Cross-entropy (nats)  | 1.428 | **1.633**  |
| Perplexity            | 4.17  | **5.12**   |
| Bits / character      | 2.06  | **2.36**   |

- **Baseline:** initial loss ≈ 4.21, matching the uniform prior over 65 symbols
  (`ln 65 ≈ 4.17`, perplexity 65). Final validation perplexity of 5.12 means the
  model is effectively choosing among ~5 characters per step.
- **Curves:** loss falls steeply through warmup and the first ~1,000 steps, then
  the train/val curves separate slightly and both flatten — see
  [`figures/loss_curve.png`](figures/loss_curve.png) and the LR schedule in
  [`figures/lr_schedule.png`](figures/lr_schedule.png).
- **Qualitative:** generated text reproduces the play format — caps speaker
  labels, colons, line breaks, apostrophes, and mostly valid English words —
  without word-level supervision. It is not grammatical at the sentence level,
  which is expected for a sub-1M-parameter character model.

## 6. Tradeoffs & decisions

- **Best-by-validation checkpointing.** Both `best.pt` and `final.pt` are
  written; `best.pt` is the lowest-val-loss snapshot so a late-training drift
  never costs the best model. The reported numbers are from `best.pt`.
- **History stored inside the checkpoint.** Loss/LR history is serialized into
  the `.pt` rather than a separate log file, so `evaluate.py` can re-plot curves
  from a checkpoint alone — there's one artifact to move around, not two.
- **Contiguous (not shuffled) val split.** Honest held-out evaluation matters
  more than squeezing out a slightly lower number; shuffled windows would leak.
- **Fail-loud device resolution.** Requesting `cuda` when it's unavailable
  raises instead of silently downgrading, so a misconfigured run is obvious.
- **Deterministic, seeded RNG** for batch sampling and evaluation, with a
  dedicated generator decoupled from global RNG, so runs and eval estimates
  are reproducible.

**Limitations.** Capacity is tiny by design, so output is locally fluent but not
globally coherent. The model only knows the 65 characters it was trained on —
generation with an out-of-vocabulary prompt character raises a clear `KeyError`.
There is no KV cache, so generation recomputes attention over the whole context
each step; fine at `T = 128` but quadratic in context length.

## 7. How to run

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py        # fetch corpus -> data/input.txt
python scripts/explore_data.py         # corpus stats

# train (CPU)
python scripts/train.py --config configs/default.yaml --device cpu
#   -> checkpoints/best.pt and checkpoints/final.pt

# generate
python scripts/generate.py --checkpoint checkpoints/best.pt \
    --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8 --top-k 40

# evaluate + plot curves
python scripts/evaluate.py --checkpoint checkpoints/best.pt --reports-dir figures

# tests
python -m pytest
```

CLI flags (`--max-steps`, `--warmup-steps`, `--batch-size`, `--device`,
`--out-dir`, `--data-path`) override the YAML without editing it — handy for
smoke runs.

## 8. How to extend

- **Scale up.** Bump `n_layer`, `n_embd`, and `block_size` in
  `configs/default.yaml`; the code is shape-agnostic. On a GPU, set
  `device: auto`.
- **KV cache.** Cache per-layer keys/values in `generate` to make sampling
  linear instead of quadratic in context length.
- **Weight tying.** Share `token_embedding.weight` with `head.weight` (GPT-2
  does this) to cut parameters and often improve loss.
- **New corpus.** Point `data.data_path` at any UTF-8 text file — the tokenizer
  rebuilds its vocabulary automatically. Try code, lyrics, or another author.
- **Better sampling.** Add nucleus (top-p) sampling alongside top-k in
  `GPT.generate`.

## 9. References

- Vaswani et al., *Attention Is All You Need* (2017) — the Transformer.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2,
  2019) — pre-norm decoder stack and the weight-init scheme used here.
- A. Karpathy, *nanoGPT* / *minGPT* and the Tiny Shakespeare corpus — reference
  point for the char-LM setup.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW, 2019) and
  *SGDR* (cosine annealing, 2017).
- PyTorch (`torch`), Matplotlib, NumPy, PyYAML.
