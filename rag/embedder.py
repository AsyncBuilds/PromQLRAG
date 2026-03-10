import time
import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus_client import MetricMeta

logger = logging.getLogger(__name__)

# model all-MiniLM-L6-v2 is tiny but good enough for metric names
# can be overridden via EMBEDDER_MODEL env var if needed
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class MetricEmbedder:

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None          # lazy load on first use
        self._index: np.ndarray = None
        self._names: list[str] = []
        self._index_ts: float = 0.0

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("embedding model loaded")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def build_index(self, metrics: "dict[str, MetricMeta]"):
        self._load_model()

        names = list(metrics.keys())
        texts = []
        for name in names:
            meta = metrics[name]
            if meta.help:
                texts.append(f"{name}: {meta.help}")
            else:
                # if no help text, then use name with underscores as spaces
                # so "http_request_duration_seconds" becomes
                # "http request duration seconds" which tokenizes better
                texts.append(name.replace("_", " "))

        t0 = time.time()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        elapsed = time.time() - t0
        logger.info(f"built embedding index: {len(names)} metrics in {elapsed:.2f}s")

        self._index = np.array(embeddings, dtype=np.float32)
        self._names = names
        self._index_ts = time.time()

    def search(
        self,
        query: str,
        top_k: int = 10,
        service_hints: list[str] = None,
        metrics: "dict[str, MetricMeta]" = None,
    ) -> list["MetricMeta"]:
        if self._index is None or len(self._names) == 0:
            logger.warning("embedding index not built, returning empty results")
            return []

        self._load_model()

        q_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        # cosine similarity = dot product (both L2-normalized)
        scores = self._index @ q_vec

        # get top candidates (fetch more than top_k to allow service filtering)
        fetch_k = min(top_k * 3, len(self._names))
        top_indices = np.argpartition(scores, -fetch_k)[-fetch_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        candidates = [
            (self._names[i], float(scores[i]))
            for i in top_indices
        ]

        if not metrics:
            return [type("MetricMeta", (), {"name": n})() for n, _ in candidates[:top_k]]

        # apply service hint preference:
        # metrics with a matching label value get a score boost
        # this is additive on top of semantic score, not a hard filter
        # so if no metrics match the service hint we still return good results
        service_hints_lower = [s.lower() for s in (service_hints or [])]

        def service_boost(name: str) -> float:
            if not service_hints_lower or name not in metrics:
                return 0.0
            meta = metrics[name]
            all_values = {
                v.lower()
                for vals in meta.labels.values()
                for v in vals
            }
            svc_values = {v.lower() for v in meta.labels.get("service", [])}
            boost = 0.0
            for hint in service_hints_lower:
                if hint in svc_values:
                    boost += 0.15   # strong boost: matches 'service' label
                elif hint in all_values:
                    boost += 0.08   # medium boost: matches any label
            return boost

        # re-rank with service boost
        reranked = sorted(
            candidates,
            key=lambda x: x[1] + service_boost(x[0]),
            reverse=True,
        )

        # filter out internal-only metrics when app metrics are present
        _INTERNAL_JOBS = {"prometheus", "node"}
        results = []
        has_app = False
        for name, _ in reranked[:top_k * 2]:
            if name not in metrics:
                continue
            meta = metrics[name]
            job_vals = set(meta.labels.get("job", []))
            if job_vals and not job_vals.issubset(_INTERNAL_JOBS):
                has_app = True
                break

        for name, _ in reranked:
            if name not in metrics:
                continue
            meta = metrics[name]
            if has_app:
                job_vals = set(meta.labels.get("job", []))
                if job_vals and job_vals.issubset(_INTERNAL_JOBS):
                    continue
            results.append(meta)
            if len(results) >= top_k:
                break

        return results

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._names) > 0