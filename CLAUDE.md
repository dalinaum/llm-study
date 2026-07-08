# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal LLM study project working through tokenization concepts (in the style of "Build a Large Language Model (From Scratch)"). It consists of standalone, numbered scripts that build on each other conceptually — from raw text splitting to a hand-rolled tokenizer to GPT-2 BPE via tiktoken. Scripts print their results; there are no tests. Comments and print output are partly in Korean.

## Commands

Uses `uv` for dependency and environment management (Python 3.13+, see `.python-version`).

```bash
uv run python 5_bpe.py     # run any script (each is a standalone entry point)
uv sync                    # install dependencies (tiktoken)
```

The default VS Code build task (`.vscode/tasks.json`) runs the current file with `uv run python`.

## Structure

- Numbered scripts (`1_open_verdict.py` … `5_bpe.py`) are sequential study steps: reading `the-verdict.txt`, regex-based token splitting, building a vocab, using `SimpleTokenizerV1/V2`, handling special tokens (`<|endoftext|>`, `<|unk|>`), and BPE with tiktoken.
- `simple_tokenizer.py` is the only shared module: `SimpleTokenizerV1` (basic vocab lookup) and `SimpleTokenizerV2` (adds `<|unk|>` fallback and `<|endoftext|>` handling).
- `the-verdict.txt` is the training/sample text all scripts read from the repo root, so scripts must be run from the repo root.

When adding a new study step, follow the existing convention: a new numbered script at the root that can be run standalone.
