import os
import logging
from dataclasses import dataclass
from prometheus_client import PrometheusClient, MetricMeta
from embedder import MetricEmbedder
from extractor import extract, ExtractedEntities
from context_builder import build_context, build_prompt
from backends import get_backend

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    query: str
    promql: str
    context: str
    retrieved_metrics: list[str]
    entities: dict
    error: str = ""


class PromQLRAG:
    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        backend: str = "ollama",
        model: str = None,
        api_key: str = None,
        top_k: int = 8,
        cache_ttl: int = 60,
        embedder_model: str = None,
    ):
        self.prom    = PrometheusClient(url=prometheus_url, ttl=cache_ttl)
        self.backend = get_backend(backend, model=model, api_key=api_key)
        self.top_k   = top_k

        _emb_model = embedder_model or os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")
        self.embedder = MetricEmbedder(model_name=_emb_model)

        # every time the metric cache refreshes, the embedding index is rebuilt as well
        self.prom._on_refresh = self.embedder.build_index

    # Full RAG pipeline: query -> extract hints -> semantic search -> build context -> generate PromQL
    def generate(self, query: str) -> RAGResult:
        metrics = self.prom.get_metrics()

        if not self.embedder.is_ready:
            self.embedder.build_index(metrics)

        entities = extract(query)

        retrieved = self.embedder.search(
            query=query,
            top_k=self.top_k,
            service_hints=entities.service_hints,
            metrics=metrics,
        )

        context = build_context(retrieved, entities)

        prompt = build_prompt(query, context)
        try:
            promql = self.backend.generate(prompt)
        except Exception as e:
            return RAGResult(
                query=query,
                promql="",
                context=context,
                retrieved_metrics=[m.name for m in retrieved],
                entities=_entities_to_dict(entities),
                error=str(e),
            )

        return RAGResult(
            query=query,
            promql=promql,
            context=context,
            retrieved_metrics=[m.name for m in retrieved],
            entities=_entities_to_dict(entities),
        )

    def health(self) -> dict:
        return {
            "prometheus":     self.prom.is_healthy(),
            "model":          self.backend.is_available(),
            "cached_metrics": len(self.prom._cache) if self.prom._cache else 0,
            "embedder_ready": self.embedder.is_ready,
        }


def _entities_to_dict(e: ExtractedEntities) -> dict:
    return {
        "keywords":         e.keywords,
        "resources":        e.resources,
        "service_hints":    e.service_hints,
        "aggregation_hint": e.aggregation_hint,
        "time_window":      e.time_window,
        "quantile":         e.quantile,
    }