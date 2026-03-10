# Benchmark

Runs 25 queries against our RAG and [prometheus-rag](https://github.com/machadovilaca/prometheus-rag) using the same model (gpt-4o-mini), so the only variable is retrieval quality.

## Setup

**1. Start the test stack**

```bash
cd ../docker
docker compose up -d
```

Check http://localhost:9090/targets, all three targets should be UP before running anything.

**2. Start our RAG**

```bash
cd ../rag
pip install -r requirements.txt

PROMETHEUS_URL=http://localhost:9090 \
RAG_BACKEND=openai \
RAG_MODEL=gpt-4o-mini \
OPENAI_API_KEY=sk-... \
python api.py
```

**3. Start prometheus-rag**

```bash
git clone https://github.com/machadovilaca/prometheus-rag
cd prometheus-rag
cp .env.example .env
```

Edit `.env`:
```
PRAG_PROMETHEUS_URL=http://localhost:9090
PRAG_LLM_BASE_URL=https://api.openai.com/v1/
PRAG_LLM_API_KEY=sk-...
PRAG_LLM_MODEL=gpt-4o-mini
PRAG_PORT=8080
```

```bash
source .env && go run main.go  # requires Go 1.22+
```

prometheus-rag indexes all metrics at startup (~2 min). Wait for the indexing log before running the benchmark.

**4. Run**

```bash
pip install requests

python run_benchmark.py \
  --our-url http://localhost:8082 \
  --prag-url http://localhost:8080 \
  --openai-key sk-... \
  --output results.json
```

## Scoring

Each query is scored on four dimensions, averaged for a total 0.0–1.0:

| dimension | check |
|---|---|
| valid_promql | Prometheus can parse it without a syntax error |
| correct_metrics | uses at least one of the expected metric names |
| correct_functions | uses the expected functions (rate, histogram_quantile, etc.) |
| executes | returns actual data from the test Prometheus |

`correct_functions` gives 0.5 partial credit if some but not all expected functions are present.

## Queries

25 queries across six categories (counter, gauge, histogram, rate, ratio, topk) at three difficulty levels. See `queries.json` for the full list with expected metrics and functions.