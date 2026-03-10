import time
import requests
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricMeta:
    name: str
    help: str = ""
    type: str = ""
    labels: dict[str, list[str]] = field(default_factory=dict)


class PrometheusClient:
    def __init__(self, url: str = "http://localhost:9090", ttl: int = 60):
        self.url = url.rstrip("/")
        self.ttl = ttl
        self._cache: Optional[dict[str, MetricMeta]] = None
        self._cache_ts: float = 0
        self._bucket_bases: set[str] = set()
        self._on_refresh = None  # called with metrics dict after each cache refresh

    def get_metrics(self) -> dict[str, MetricMeta]:
        if self._cache is None or (time.time() - self._cache_ts) > self.ttl:
            self._refresh()
        return self._cache

    def get_metric(self, name: str) -> Optional[MetricMeta]:
        return self.get_metrics().get(name)

    def is_healthy(self) -> bool:
        try:
            return requests.get(f"{self.url}/-/healthy", timeout=3).status_code == 200
        except Exception:
            return False

    def _refresh(self):
        names    = self._fetch_names()
        metadata = self._fetch_metadata()

        metrics = {}
        for name in names:
            meta  = metadata.get(name, {})
            mtype = meta.get("type", "")
            if name in self._bucket_bases:
                mtype = "histogram"
            metrics[name] = MetricMeta(
                name=name,
                help=meta.get("help", ""),
                type=mtype,
            )

        for name, metric in metrics.items():
            metric.labels = self._fetch_labels(name)

        self._cache    = metrics
        self._cache_ts = time.time()

        if self._on_refresh:
            self._on_refresh(metrics)

    def _fetch_names(self) -> list[str]:
        try:
            resp = requests.get(
                f"{self.url}/api/v1/label/__name__/values", timeout=10
            )
            resp.raise_for_status()
            names = resp.json().get("data", [])
        except Exception as e:
            raise RuntimeError(f"failed to fetch metric names: {e}")

        self._bucket_bases = {
            n[:-len("_bucket")] for n in names if n.endswith("_bucket")
        }

        # collapse histogram series to their base name, drop internal suffixes
        filtered, seen = [], set()
        for n in names:
            if n.endswith(("_created", "_count", "_sum")):
                continue
            if n.endswith("_bucket"):
                n = n[:-len("_bucket")]
            if n not in seen:
                seen.add(n)
                filtered.append(n)

        return filtered

    def _fetch_metadata(self) -> dict[str, dict]:
        try:
            resp = requests.get(f"{self.url}/api/v1/metadata", timeout=10)
            resp.raise_for_status()
            raw = resp.json().get("data", {})
            return {
                name: {"help": e[0].get("help", ""), "type": e[0].get("type", "")}
                for name, e in raw.items() if e
            }
        except Exception:
            return {}

    def _fetch_labels(self, metric_name: str) -> dict[str, list[str]]:
        try:
            resp = requests.get(
                f"{self.url}/api/v1/series",
                params={"match[]": metric_name, "limit": "50"},
                timeout=5,
            )
            resp.raise_for_status()
            series = resp.json().get("data", [])
        except Exception:
            return {}

        labels: dict[str, set] = {}
        for s in series:
            for k, v in s.items():
                if k != "__name__":
                    labels.setdefault(k, set()).add(v)

        return {k: sorted(v)[:20] for k, v in labels.items()}