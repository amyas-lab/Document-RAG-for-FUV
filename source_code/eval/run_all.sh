#!/usr/bin/env bash
# Reproduce every Task 2 experiment. Run from source_code/ with the `rag` environment active
# and Ollama serving the models below. Takes about 1 hour on an Apple M2 (16 GB).
#
#   ollama pull nomic-embed-text mxbai-embed-large qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b llama3.2:3b
#   bash eval/run_all.sh
set -euo pipefail
export QDRANT_PATH=${QDRANT_PATH:-data/qdrant_eval}   # separate index so a running chat session is not locked

# 1) Retrieval ablation: parser x chunking x embedding x retrieval mode (no LLM)
python -m eval.run_eval retrieval

# 2) LLM comparison on the best retrieval configuration (strict prompt)
python -m eval.run_eval answers --models qwen2.5:0.5b,qwen2.5:1.5b,qwen2.5:3b,llama3.2:3b

# 3) Prompt ablation: baseline "fallback" prompt vs strict prompt
python -m eval.run_eval answers --models qwen2.5:3b --mode fallback

# 4) Pipeline ablation: baseline-like pipeline (pypdf + fixed chunks + dense) with the same LLM
python -m eval.run_eval answers --models qwen2.5:3b --parser pypdf --chunking fixed --retrieval dense

python -m eval.run_eval report
