import re
from dataclasses import dataclass, field


AGGREGATION_HINTS = {
    "total":      "sum",
    " sum ":      "sum",
    "average":    "avg",
    " avg ":      "avg",
    "mean":       "avg",
    " max ":      "max",
    "maximum":    "max",
    "peak":       "max",
    " min ":      "min",
    "minimum":    "min",
    " count ":    "count",
    " rate ":     "rate",
    "per second": "rate",
    "increase":   "increase",
    "percentile": "histogram_quantile",
    "p99":        "histogram_quantile",
    "p95":        "histogram_quantile",
    "p50":        "histogram_quantile",
    "quantile":   "histogram_quantile",
    " top ":      "topk",
    " bottom ":   "bottomk",
    "slowest":    "topk",
    "fastest":    "bottomk",
}

_TIME_PATTERN = re.compile(
    r"\b(\d+)\s*(second|minute|hour|day|week|s|m|h|d|w)s?\b",
    re.IGNORECASE,
)
_TIME_UNITS = {
    "second": "s", "minute": "m", "hour": "h", "day": "d", "week": "w",
    "s": "s", "m": "m", "h": "h", "d": "d", "w": "w",
}

# patterns for pulling service/job names out of the query
# used to prefer metrics with a matching label value during retrieval
_SERVICE_PATTERNS = [
    r"for\s+(?:the\s+)?([\w][\w-]*)\s+service",
    r"([\w][\w-]*)\s+service",
    r"([\w][\w-]*)\s+pod",
    r"([\w][\w-]*)\s+container",
    r"job\s*=\s*[\"']?([\w][\w-]*)[\"']?",
    r"service\s*=\s*[\"']?([\w][\w-]*)[\"']?",
    r"for\s+(?:the\s+)?([\w][\w-]*)\b",
]

_STOPWORDS = {
    "the", "a", "an", "my", "our", "all", "each", "any",
    "per", "by", "of", "in", "for", "on", "at", "is",
    "last", "past", "over", "this", "that", "with",
}


@dataclass
class ExtractedEntities:
    service_hints: list[str]  = field(default_factory=list)
    aggregation_hint: str     = ""
    time_window: str          = ""
    quantile: float           = 0.0
    keywords: list[str]       = field(default_factory=list)  # debug only
    raw: str                  = ""


def extract(query: str) -> ExtractedEntities:
    lower = query.lower()
    e = ExtractedEntities(raw=query)

    m = _TIME_PATTERN.search(lower)
    if m:
        e.time_window = f"{m.group(1)}{_TIME_UNITS.get(m.group(2).lower(), 'm')}"

    q = re.search(r"\bp(\d+)\b", lower)
    if q:
        e.quantile = int(q.group(1)) / 100.0
    q = re.search(r"(\d+)(?:th|st|nd|rd)?\s*percentile", lower)
    if q:
        e.quantile = int(q.group(1)) / 100.0

    for phrase, agg in AGGREGATION_HINTS.items():
        if phrase in lower:
            e.aggregation_hint = agg
            break

    hints = set()
    for pattern in _SERVICE_PATTERNS:
        for match in re.finditer(pattern, lower):
            hint = match.group(1)
            if hint not in _STOPWORDS:
                hints.add(hint)
    e.service_hints = sorted(hints)

    _kw_stopwords = {
        "show", "give", "find", "get", "what", "which", "where", "when",
        "how", "many", "much", "with", "that", "this", "from", "over",
        "last", "past", "the", "and", "for", "are", "has", "have", "been",
        "each", "per", "all", "any", "its", "by", "of", "in", "on", "at",
        "to", "is", "be", "do", "not", "rate", "time", "metric", "query",
        "across", "between", "using", "into", "than", "more", "less",
        "number", "count", "total", "average", "current", "latest",
    }
    e.keywords = sorted(
        w for w in re.findall(r"\b[a-z]\w+\b", lower)
        if len(w) > 3 and w not in _kw_stopwords
    )

    return e