# Docker deployment

Four services, started in dependency order by `docker compose`:

| Service | Image | Port | Purpose |
|---|---|---|---|
| `ollama` | `ollama/ollama:0.40.13` | 11434 | LLM + embedding server (models in the `ollama_models` volume) |
| `ollama-init` | `ollama/ollama:0.40.13` | – | One-shot job: pulls `LLM_MODEL` and `EMBED_MODEL`, then exits |
| `qdrant` | `qdrant/qdrant:v1.20.0` | 6333 | Vector database (hybrid dense + BM25), dashboard at `/dashboard` |
| `api` | built from `Dockerfile.api` | 8000 | FastAPI backend, Swagger UI at `/docs`. Builds the index on start |
| `ui` | built from `Dockerfile.ui` (`nginx:1.31.8-alpine`) | 8080 | Static web UI; proxies `/api/*` to the API |

```
ollama ──healthy──► ollama-init ──completed──┐
qdrant ──healthy─────────────────────────────┴──► api ──healthy──► ui
```

The handbook PDF is mounted read-only from `source_code/data/documents/`. Data that has to survive
restarts lives in named volumes: `qdrant_storage` holds the index, `ollama_models` holds about 3 GB of models.

## Requirements

- Docker Desktop 4.x (macOS/Windows) or Docker Engine 24+ with Compose v2.20+ (Linux).
- About 8 GB RAM for Docker and about 6 GB of disk (images plus models).
- The first start downloads about 2.3 GB of models (`qwen2.5:3b` + `nomic-embed-text`).

## 1. Build and run (fully containerised)

```bash
cd docker
docker compose up --build -d        # first run: about 5-10 min (image build + model pull + indexing)
docker compose logs -f ollama-init api   # watch model pulls and indexing
```

Open the UI at <http://localhost:8080>, the API docs at <http://localhost:8000/docs> and the
Qdrant dashboard at <http://localhost:6333/dashboard>.

> **Apple Silicon note.** Containers cannot use the Metal GPU, so generation runs on CPU
> (measured on an M2: about 38 s per answer vs about 14 s with host Ollama). For demos, prefer the host-Ollama mode below.

## 2. Host-Ollama mode (GPU on the host)

Run Ollama natively on the host, and keep Qdrant, the API and the UI in containers:

```bash
ollama pull qwen2.5:3b && ollama pull nomic-embed-text      # on the host
cd docker
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build -d
```

The override moves `ollama` and `ollama-init` into an inactive profile and points the API at
`http://host.docker.internal:11434`. On Linux, start the host server with `OLLAMA_HOST=0.0.0.0 ollama serve`.

## 3. Configuration

Compose reads these variables from the shell, or from a `docker/.env` file. That file is gitignored, and `../.env.example` shows the format.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:3b` | Default chat model (the UI can switch to any installed model) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model; changing it creates a new collection |
| `EXTRA_MODELS` | – | Extra models for `ollama-init` to pull, e.g. `"qwen2.5:1.5b llama3.2:3b"` |
| `TOP_K` | `5` | Retrieved excerpts per question |
| `PROMPT_MODE` | `strict` | `strict` (handbook only) or `fallback` (baseline prompt) |

Example: `LLM_MODEL=llama3.2:3b EXTRA_MODELS="qwen2.5:1.5b" docker compose up -d`.

## 4. Test

```bash
docker compose ps                                     # all services "healthy" (ollama-init "exited (0)")
curl -s localhost:8000/health | python -m json.tool  # {"status": "ok", "n_chunks": 86, ...}
curl -s localhost:8000/models
curl -s -X POST localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"question": "What are the requirements for a minor in Applied Mathematics?"}' | python -m json.tool
curl -N -X POST localhost:8000/chat/stream -H 'Content-Type: application/json' \
     -d '{"question": "What are the prerequisites for MATH 305?"}'          # SSE stream
curl -s -X POST localhost:8000/index -H 'Content-Type: application/json' -d '{"force": false}'
```

## 5. Update documents

Put PDF or TXT files in `source_code/data/documents/` and then call `POST /index`, or click **Rebuild index**
in the UI sidebar. The API also re-indexes on restart whenever the documents or the chunking settings change,
because a content hash is stored with every point.

## 6. Stop and clean up

```bash
docker compose stop                    # stop, keep containers
docker compose down                    # remove containers + network (volumes kept)
docker compose down -v                 # also delete the index and the downloaded models
docker image rm fuv-math-rag-api fuv-math-rag-ui
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `api` stays `starting` for a long time | First start embeds every chunk; follow `docker compose logs -f api` |
| `ollama-init` fails to pull | Check internet access; re-run with `docker compose up ollama-init` |
| `model "…" not found` in answers | The UI picked a model that is not pulled; add it to `EXTRA_MODELS` |
| Host mode: `Connection refused` to Ollama | Ollama is not running on the host, or on Linux it is bound to 127.0.0.1 only |
| Port already in use | Stop the local `uvicorn`/`ollama`, or change the left side of `ports:` |
