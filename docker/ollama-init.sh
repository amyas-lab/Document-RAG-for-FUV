#!/bin/sh
# One-shot job: pull the models the API needs into the shared `ollama_models` volume.
# Runs in the ollama/ollama image with OLLAMA_HOST pointing at the `ollama` service.
set -eu
for model in ${LLM_MODEL:-qwen2.5:3b} ${EMBED_MODEL:-nomic-embed-text} ${EXTRA_MODELS:-}; do
  if ollama show "$model" >/dev/null 2>&1; then
    echo "[ollama-init] $model already present"
  else
    echo "[ollama-init] pulling $model ..."
    ollama pull "$model"
  fi
done
echo "[ollama-init] done"
