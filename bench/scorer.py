import re
import requests
from dataclasses import dataclass


@dataclass
class Score:
    valid_promql: float      = 0.0
    correct_metrics: float   = 0.0
    correct_functions: float = 0.0
    executes: float          = 0.0
    error: str               = ""

    @property
    def total(self) -> float:
        return (self.valid_promql + self.correct_metrics +
                self.correct_functions + self.executes) / 4.0

    def to_dict(self) -> dict:
        return {
            "valid_promql":      self.valid_promql,
            "correct_metrics":   self.correct_metrics,
            "correct_functions": self.correct_functions,
            "executes":          self.executes,
            "total":             round(self.total, 3),
            "error":             self.error,
        }


def score(
    promql: str,
    expected_metrics: list[str],
    expected_functions: list[str],
    prometheus_url: str = "http://localhost:9090",
) -> Score:
    if not promql or not promql.strip():
        return Score(error="empty query")

    s = Score()

    try:
        resp = requests.get(
            f"{prometheus_url}/api/v1/query",
            params={"query": promql, "time": "1"},
            timeout=5,
        )
        data = resp.json()
        if data.get("status") == "success":
            s.valid_promql = 1.0
            if data.get("data", {}).get("result"):
                s.executes = 1.0
        elif "parse" in data.get("error", "").lower() or "syntax" in data.get("error", "").lower():
            s.error = data.get("error", "parse error")
        else:
            s.valid_promql = 1.0
            s.error = data.get("error", "")
    except Exception as e:
        s.error = str(e)
        return s

    if expected_metrics:
        used  = {_base(m) for m in _extract_metric_names(promql)}
        expected = {_base(m) for m in expected_metrics}
        s.correct_metrics = 1.0 if used & expected else 0.0
    else:
        s.correct_metrics = 1.0

    if expected_functions:
        used_fns = _extract_functions(promql)
        matches  = [f for f in expected_functions if f.lower() in used_fns]
        if len(matches) == len(expected_functions):
            s.correct_functions = 1.0
        elif matches:
            s.correct_functions = 0.5
    else:
        s.correct_functions = 1.0

    return s


def _base(name: str) -> str:
    for suffix in ("_bucket", "_total", "_count", "_sum", "_created"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def _extract_metric_names(promql: str) -> set[str]:
    cleaned = re.sub(r'"[^"]*"', '""', promql)
    cleaned = re.sub(r"'[^']*'", "''", cleaned)
    tokens  = re.findall(r'\b([a-zA-Z_:][a-zA-Z0-9_:]*)\b(?!\s*\()', cleaned)
    _skip   = {
        "by", "without", "on", "ignoring", "group_left", "group_right",
        "offset", "bool", "and", "or", "unless", "inf", "nan",
        "job", "instance", "service", "method", "handler", "status_code",
        "code", "database", "operation", "status", "le", "quantile",
        "payment_method", "error_type", "replica", "version", "environment",
        "device", "mountpoint", "fstype", "cpu", "mode", "interface",
    }
    return {t for t in tokens if t not in _skip and not t.isdigit()}


def _extract_functions(promql: str) -> set[str]:
    tokens    = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', promql)
    _not_fns  = {"by", "without", "on", "ignoring", "group_left", "group_right"}
    return {t.lower() for t in tokens if t.lower() not in _not_fns}