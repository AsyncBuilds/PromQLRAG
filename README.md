# PromQL RAG

Generates PromQL from natural language by querying your actual Prometheus instance at inference time. Instead of a static knowledge base, it fetches live metric names, types, and label values so it knows about your specific metrics, not just common ones.

Built as a companion to [AsyncBuilds/qwen3-1.7b-promql](https://huggingface.co/AsyncBuilds/qwen3-1.7b-promql) but works with any OpenAI-compatible model.

---

## How it works

Three stages at query time:

**1. Semantic retrieval** - embeds the user query with `all-MiniLM-L6-v2` and finds the most similar metrics by cosine similarity against a pre-built index of all metric names + help text. For example, "latency" finds `request_duration_seconds` even if those words don't overlap.

**2. Label hydration** - for the top-k retrieved metrics, fetches current label names and values from Prometheus live. The context reflects the actual state of your Prometheus, including new services, new pods, and label changes since the last deploy.

**3. Context injection** - formats metrics with their types, live label values, and extracted hints (time window, quantile, service filter) into a structured block the LLM can reason over. It sees your real metric names and label values, not generic textbook examples.

```
query: "p99 latency for the payment service over 5m"
         |
         v
   embed query -> cosine sim -> top-k metrics
         |
         v
   fetch live labels from Prometheus
         |
         v
   context:
     payment_processing_duration_seconds (histogram)
       Help: Payment processing latency
       Labels: service=[payment], method=[card, bank], status=[ok, failed]
     hints: time_window=5m, quantile=0.99, service=payment
         |
         v
   LLM -> histogram_quantile(0.99, sum(rate(
             payment_processing_duration_seconds_bucket{
               service="payment"}[5m])) by (le))
```

The embedding index rebuilds automatically on the same TTL as the metadata cache (default 60s). Startup takes about 1 second for a typical Prometheus with a few hundred metrics.

---

## Benchmark

Compared against [prometheus-rag](https://github.com/machadovilaca/prometheus-rag) using the same model (gpt-4o-mini) on 25 queries across a test Prometheus stack with 19 custom application metrics.

| metric | ours | prometheus-rag |
|---|---|---|
| **total score** | **0.727** | 0.676 |
| correct_metrics | **0.909** | 0.765 |
| correct_functions | **1.000** | 0.941 |
| histogram queries | **0.750** | 0.750 |
| rate queries | **0.667** | 0.607 |
| hard queries | **0.700** | 0.542 |
| avg latency | 1.25s | 1.37s |
| startup time | ~1s | ~2 min |

The main gap is `correct_metrics` at 90.9% vs 76.5%. Semantic retrieval finds the right metric family even when the user's words don't appear literally in the metric name. prometheus-rag uses LaBSE embeddings over individual series (metric + label combinations), which works but is slow to build and gets expensive at high label cardinality.

See [bench/](bench/) for the full benchmark setup, queries, and scoring methodology.

---

## Tradeoffs

Works well for:
- Single-tenant or small multi-tenant Prometheus
- Kubernetes environments where label values change frequently (live hydration keeps the context current)
- Custom application metrics with unusual naming conventions, since semantic retrieval generalizes without needing to know your naming scheme upfront

Not a great fit for:
- Very large multi-tenant Prometheus with thousands of distinct metric names and heavy concurrent load. The live label hydration (N API calls per request) becomes a bottleneck. You'd want background pre-fetching and a longer label cache TTL instead
- Non-English metric help text. MiniLM is English-only. prometheus-rag uses LaBSE which handles multilingual text

---

## Setup

```bash
cd rag
pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `requests`, `openai`, `sentence-transformers`, `numpy`

The first run downloads `all-MiniLM-L6-v2` (~22MB) from HuggingFace.

---

## Running

```bash
# with OpenAI
PROMETHEUS_URL=http://localhost:9090 \
RAG_BACKEND=openai \
RAG_MODEL=gpt-4o-mini \
OPENAI_API_KEY=sk-... \
python api.py

# with Ollama (fine-tuned model or any local model)
PROMETHEUS_URL=http://localhost:9090 \
RAG_BACKEND=ollama \
RAG_MODEL=promql \
python api.py
```

Runs on port 8082 by default (`RAG_PORT` to override).

---

## API

**POST /generate**

```bash
curl -X POST http://localhost:8082/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "p99 latency for the payment service over 5 minutes"}'
```

```json
{
  "promql": "histogram_quantile(0.99, sum(rate(payment_processing_duration_seconds_bucket{service=\"payment\"}[5m])) by (le))",
  "retrieved_metrics": ["payment_processing_duration_seconds", "payment_transactions_total"],
  "context": "Available metrics:\n- payment_processing_duration_seconds (histogram)...",
  "entities": {
    "service_hints": ["payment"],
    "time_window": "5m",
    "quantile": 0.99,
    "aggregation_hint": "histogram_quantile"
  },
  "error": ""
}
```

**GET /health**

```bash
curl http://localhost:8082/health
# {"prometheus": true, "model": true, "cached_metrics": 481, "embedder_ready": true}
```

**GET /metrics/search?q=payment+latency**

Debug endpoint that returns the top metrics the embedder retrieves for a query. Useful for checking retrieval quality before worrying about generation quality.

**POST /cache/refresh**

Forces a cache + embedding index rebuild. Useful if you've just deployed new metrics and don't want to wait for the TTL.

---

## Environment variables

| variable | default | description |
|---|---|---|
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus base URL |
| `RAG_BACKEND` | `ollama` | `ollama` or `openai` |
| `RAG_MODEL` | `promql-model` | model name passed to the backend |
| `OPENAI_API_KEY` | (none) | required for `openai` backend |
| `RAG_TOP_K` | `8` | number of metrics to retrieve per query |
| `RAG_CACHE_TTL` | `60` | metric cache TTL in seconds |
| `RAG_PORT` | `8082` | port to listen on |
| `EMBEDDER_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name |

---

## Test stack

`docker/` has a minimal Prometheus setup for local testing: Prometheus, node_exporter, and a fake app that emits 19 metrics across three simulated services (api-gateway, payment, database).

```bash
cd docker
docker compose up -d
# Prometheus UI: http://localhost:9090
# check targets: http://localhost:9090/targets
```

---

## Code structure

```
rag/
  embedder.py          # builds + queries the embedding index
  prometheus_client.py # fetches metric metadata + labels from Prometheus
  extractor.py         # pulls time window, quantile, service hints from query
  context_builder.py   # formats metrics + hints into LLM context
  rag.py               # wires the pipeline together
  api.py               # FastAPI server
  backends.py          # Ollama + OpenAI backends

bench/
  run_benchmark.py     # runs queries against both systems, prints results
  scorer.py            # scores generated PromQL against expected answers
  queries.json         # 25 test queries with expected metrics + functions

docker/
  docker-compose.yml
  prometheus.yml
  app_metrics/         # fake metrics generator
```