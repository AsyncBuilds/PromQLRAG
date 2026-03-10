from extractor import ExtractedEntities
from prometheus_client import MetricMeta


def build_context(
    metrics: list[MetricMeta],
    entities: ExtractedEntities,
    max_label_values: int = 8,
) -> str:
    """
    Format retrieved metrics into a context block for the LLM prompt.

    Example output:
        Available metrics:
        - http_requests_total (counter)
          Help: Total HTTP requests
          Labels: service=[api-gateway, payment], method=[GET, POST], status_code=[200, 500]

        Hints:
        - time window: 5m
        - suggested aggregation: rate
        - quantile: 0.99
    """
    if not metrics:
        return "No relevant metrics found in Prometheus."

    # detect paired metrics (hit/miss, success/failure, etc.) and annotate them
    # so the model understands they should be used together for ratio queries
    names = {m.name for m in metrics}
    pair_annotations = _find_pairs(names)

    lines = ["Available metrics:"]

    for meta in metrics:
        type_tag = f" ({meta.type})" if meta.type else ""
        # for histograms, explicitly tell the model to use the _bucket suffix
        # with histogram_quantile - this is the most common source of errors
        bucket_note = " [use name_bucket with histogram_quantile]" if meta.type == "histogram" else ""
        pair_note = f" [pair: {pair_annotations[meta.name]}]" if meta.name in pair_annotations else ""
        lines.append(f"- {meta.name}{type_tag}{bucket_note}{pair_note}")

        if meta.help:
            lines.append(f"  Help: {meta.help}")

        if meta.labels:
            # skip purely internal labels that add noise without helping the model
            skip_labels = {"instance", "scrape_job"}
            label_parts = []
            for label_name, values in sorted(meta.labels.items()):
                if label_name in skip_labels:
                    continue
                if not values:
                    label_parts.append(label_name)
                else:
                    vals = values[:max_label_values]
                    suffix = "..." if len(values) > max_label_values else ""
                    label_parts.append(f"{label_name}=[{', '.join(vals)}{suffix}]")
            if label_parts:
                lines.append(f"  Labels: {', '.join(label_parts)}")

        lines.append("")

    hints = []
    if entities.time_window:
        hints.append(f"time window: {entities.time_window}")
    if entities.aggregation_hint:
        hints.append(f"suggested aggregation: {entities.aggregation_hint}")
    if entities.quantile > 0:
        hints.append(f"quantile: {entities.quantile}")
    if entities.service_hints:
        hints.append(f"service filter hints: {', '.join(entities.service_hints)}")

    if hints:
        lines.append("Hints:")
        for h in hints:
            lines.append(f"- {h}")

    return "\n".join(lines)


# suffix pairs that indicate complementary metrics
_SUFFIX_PAIRS = [
    ("_hits_total",   "_misses_total"),
    ("_hit_total",    "_miss_total"),
    ("_success",      "_failure"),
    ("_success_total","_failure_total"),
    ("_active",       "_idle"),
]

def _find_pairs(names: set[str]) -> dict[str, str]:
    annotations = {}
    for a_suffix, b_suffix in _SUFFIX_PAIRS:
        for name in names:
            if name.endswith(a_suffix):
                partner = name[:-len(a_suffix)] + b_suffix
                if partner in names:
                    annotations[name]    = partner
                    annotations[partner] = name
    return annotations


def build_prompt(query: str, context: str) -> str:
    return (
        f"Request: {query}\n"
        f"Context: {context}\n\n"
        f"Only use the metrics listed above. Do not use any other metric names."
    )