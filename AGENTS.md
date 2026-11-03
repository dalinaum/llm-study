# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal LLM study project following "Build a Large Language Model (From Scratch)". It consists of standalone, numbered scripts that build on each other conceptually — tokenization and BPE, data loading and embeddings, attention, the GPT model, text generation and evaluation, training, loading pretrained GPT-2 weights, and preparing an SMS spam dataset for classification fine-tuning. Scripts print their results; there are no tests (`*_test.py` files are study scripts, not a test suite). Comments and print output are partly in Korean.

## Commands

Uses `uv` for dependency and environment management (Python 3.13+, see `.python-version`).

```bash
uv run python 05_bpe.py     # run any script (each is a standalone entry point)
uv sync                    # install dependencies (torch, tiktoken, numpy, pandas, matplotlib, tqdm, tensorflow)
```

The default VS Code build task (`.vscode/tasks.json`) runs the current file with `uv run python`.

## Structure

- Numbered scripts (`01_open_verdict.py` … `30_spam_dataset_test.py`) are sequential study steps: tokenization and BPE (01–05), sliding-window data loading and embeddings (06–09), attention (10–13), GPT model components and the full model (14–22), generation, sampling and evaluation (23–26), training and loading pretrained GPT-2 weights (27–29), and the SMS spam dataset (30).
- Shared modules are plain top-level `.py` files extracted from the numbered scripts and imported by later ones, e.g. `simple_tokenizer.py`, `gpt_dataset_v1.py`, `multi_head_attention.py`, `transformer_block.py`, `gpt_model.py`, `generate_util.py`, `train_util.py`, `loss_util.py`, `gpt_load_helper.py`. `gpt_download.py` is fetched from the book's companion repo by `28_download_gpt_download.py`, not hand-written.
- Data and weights are opened by relative path from the repo root, so scripts must be run from the repo root. `the-verdict.txt` (tracked) is the sample text for the early scripts; GPT-2 weights (`gpt2/`), checkpoints (`*.pth`) and the SMS spam files are downloaded or written by the scripts (`gpt2/` and `*.pth` are git-ignored).

When adding a new study step, follow the existing convention: a new numbered script at the root that can be run standalone.
