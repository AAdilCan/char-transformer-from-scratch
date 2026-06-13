# char-transformer-from-scratch

A decoder-only Transformer (GPT-style) implemented from scratch in PyTorch for
character-level text generation. No `nn.Transformer`, no Hugging Face — the
attention, multi-head wiring, positional embeddings, and training loop are all
written by hand so the internals are fully visible.

I built this to understand exactly what happens inside a GPT block: how causal
self-attention is masked, how heads are split and recombined, and how the
training/sampling loop actually works. It trains on CPU on the ~1 MB Tiny
Shakespeare corpus and generates Shakespeare-flavored text.

## Status

Work in progress — building over the week:

- [x] Day 1 — Project scaffold, config + logging, data download & exploration
- [x] Day 2 — Char tokenizer, dataset, batching
- [x] Day 3 — Attention, Transformer blocks, GPT model
- [x] Day 4 — Training loop, LR schedule, checkpointing
- [x] Day 5 — Evaluation, loss curves, sampling, perplexity
- [x] Day 6 — Test suite, edge cases, refactor
- [ ] Day 7 — Documentation, polished results

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# download the ~1 MB Tiny Shakespeare corpus
python scripts/download_data.py

# look at corpus statistics
python scripts/explore_data.py
```

## Project layout

```
src/char_transformer/   core package (config, model, tokenizer, training)
scripts/                CLI entry points (download, explore, train, generate)
configs/                YAML hyperparameter configs
tests/                  pytest suite
reports/                training curves and generated samples
```

## Why character-level?

A character vocabulary is tiny (~65 symbols for Shakespeare), so there is no
subword tokenizer to train and the embedding/output layers stay small. That
keeps the whole thing CPU-trainable while still exercising every part of a real
Transformer: embeddings, causal attention, residual blocks, and autoregressive
sampling.
