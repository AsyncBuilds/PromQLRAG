import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import PromQLRAG


PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL",   "http://localhost:9090")
BACKEND        = os.environ.get("RAG_BACKEND",      "ollama")
MODEL          = os.environ.get("RAG_MODEL",        "promql-model")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY",   "")
TOP_K          = int(os.environ.get("RAG_TOP_K",    "8"))
CACHE_TTL      = int(os.environ.get("RAG_CACHE_TTL","60"))
PORT           = int(os.environ.get("RAG_PORT",     "8082"))


app = FastAPI(title="PromQL RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = PromQLRAG(
    prometheus_url=PROMETHEUS_URL,
    backend=BACKEND,
    model=MODEL,
    api_key=OPENAI_API_KEY or None,
    top_k=TOP_K,
    cache_ttl=CACHE_TTL,
)


class GenerateRequest(BaseModel):
    query: str


class GenerateResponse(BaseModel):
    query: str
    promql: str
    retrieved_metrics: list[str]
    entities: dict
    context: str
    error: str


class MetricInfo(BaseModel):
    name: str
    type: str
    help: str
    labels: dict


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    result = rag.generate(req.query)

    if result.error:
        raise HTTPException(status_code=502, detail=result.error)

    return GenerateResponse(
        query=result.query,
        promql=result.promql,
        retrieved_metrics=result.retrieved_metrics,
        entities=result.entities,
        context=result.context,
        error=result.error,
    )


@app.get("/health")
def health():
    status = rag.health()
    if not status["prometheus"]:
        raise HTTPException(status_code=503, detail="Prometheus unreachable")
    return status


@app.get("/metrics/search", response_model=list[MetricInfo])
def search_metrics(
    q: str = Query(..., description="natural language search query"),
    top_k: int = Query(10, description="max results"),
):
    metrics = rag.embedder.search(
        query=q,
        top_k=top_k,
        metrics=rag.prom.get_metrics(),
    )
    return [
        MetricInfo(name=m.name, type=m.type, help=m.help, labels=m.labels)
        for m in metrics
    ]


@app.get("/metrics/list", response_model=list[str])
def list_metrics():
    return sorted(rag.prom.get_metrics().keys())


@app.post("/cache/refresh")
def refresh_cache():
    rag.prom._cache    = None
    rag.prom._cache_ts = 0
    metrics = rag.prom.get_metrics()
    return {"refreshed": True, "metric_count": len(metrics)}


@app.on_event("startup")
def startup():
    print(f"Prometheus : {PROMETHEUS_URL}")
    print(f"Backend    : {BACKEND} / {MODEL}")
    print(f"Top-K      : {TOP_K}")
    print(f"Cache TTL  : {CACHE_TTL}s")
    try:
        metrics = rag.prom.get_metrics()
        print(f"Loaded {len(metrics)} metrics from Prometheus")
    except Exception as e:
        print(f"Warning: could not pre-warm cache: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False)