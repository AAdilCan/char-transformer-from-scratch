# Model card — char-transformer-from-scratch

## Model details

- **Type:** decoder-only Transformer (GPT-style) language model, built from
  scratch in PyTorch.
- **Granularity:** character level, 65-symbol vocabulary.
- **Size:** 826,368 parameters — 4 layers, 4 heads, 128 embedding dim,
  128-character context (`block_size`), dropout 0.1.
- **Objective:** next-character prediction (mean cross-entropy).
- **Framework:** PyTorch 2.2; no pretrained weights, no external model code.
- **Version:** 0.1.0. Single training run, seed 1337.

## Intended use

Educational and demonstrative. It exists to show a hand-built Transformer
training end-to-end and generating text, and to be a clear, hackable reference
for the GPT block internals. Appropriate uses: learning, experimentation,
extending the architecture, generating Shakespeare-flavored sample text.

**Out of scope:** any production or user-facing text generation, factual
question answering, or any task requiring coherent multi-sentence output. This
is a sub-1M-parameter character model, not an assistant.

## Training data

Tiny Shakespeare — 1,115,394 characters of Shakespeare's works (40,000 lines,
65 unique characters). Split contiguously 90% train / 10% validation; the
validation tail is never seen during training. The model's "knowledge" is
entirely the statistics of this single corpus.

## Training procedure

- AdamW (`betas = 0.9, 0.99`), weight decay 0.1 on 2-D weights only.
- Linear warmup (200 steps) to peak LR `3e-4`, then cosine decay to `0.1 ×` peak.
- Gradient clipping at global norm 1.0.
- 5,000 steps, batch size 64, on CPU. Best-by-validation checkpoint retained.

## Evaluation

Train/val loss and perplexity averaged over 200 random batches with dropout off.

| Metric               | Train | Validation |
|----------------------|-------|------------|
| Cross-entropy (nats) | 1.428 | 1.633      |
| Perplexity           | 4.17  | 5.12       |
| Bits / character     | 2.06  | 2.36       |

Baseline (untrained) loss ≈ 4.21 ≈ `ln 65`, i.e. perplexity ≈ 65.

## Limitations and biases

- **Locally fluent, globally incoherent.** It learns spelling, the play format,
  and short phrases, but not sentence-level grammar or plot.
- **Corpus-bound.** Vocabulary and style are fixed to early-modern English
  Shakespeare; it reflects the language, themes, and biases of that text and
  nothing else. A prompt character outside the 65-symbol vocabulary raises a
  `KeyError` by design.
- **Tiny capacity** (0.83M params) — far below anything usable for real tasks.
- **No safety filtering.** Output is raw samples from the corpus distribution.

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py --config configs/default.yaml --device cpu
python scripts/evaluate.py --checkpoint checkpoints/best.pt --reports-dir figures
```
