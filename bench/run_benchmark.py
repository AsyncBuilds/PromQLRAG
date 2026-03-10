import argparse
import json
import os
import re
import time
from datetime import datetime

import requests

from scorer import score, Score


def call_our_rag(url: str, query: str, model: str, api_key: str) -> tuple[str, float]:
    t0 = time.time()
    resp = requests.post(f"{url}/generate", json={"query": query}, timeout=60)
    latency = time.time() - t0
    resp.raise_for_status()
    return resp.json()["promql"], latency


def call_prometheus_rag(url: str, query: str) -> tuple[str, float]:
    t0 = time.time()
    resp = requests.post(f"{url}/query", json={"query": query}, timeout=60)
    latency = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    promql = (
        data.get("promql") or data.get("response") or
        data.get("result") or data.get("answer") or ""
    )
    promql = re.sub(r"^```(?:promql)?\s*|\s*```$", "", promql.strip(), flags=re.MULTILINE)
    return promql.strip(), latency


def run_benchmark(queries, our_url, prag_url, prometheus_url,
                  openai_model, openai_key, delay) -> dict:
    systems = [
        ("ours_openai", "our",  openai_model),
        ("prag_openai", "prag", openai_model),
    ]

    reachable = {}
    print("checking system availability...")
    for sys_id, kind, model in systems:
        try:
            if kind == "our":
                requests.get(f"{our_url}/health", timeout=3).raise_for_status()
            else:
                requests.post(f"{prag_url}/query", json={"query": "test"}, timeout=5)
            reachable[sys_id] = True
            print(f"  {sys_id}: ok")
        except Exception as e:
            reachable[sys_id] = False
            print(f"  {sys_id}: unreachable ({e})")

    results = {
        "meta": {
            "timestamp":      datetime.utcnow().isoformat(),
            "our_url":        our_url,
            "prag_url":       prag_url,
            "prometheus_url": prometheus_url,
            "model":          openai_model,
            "query_count":    len(queries),
        },
        "systems": {s[0]: [] for s in systems},
    }

    total = len(queries) * len(systems)
    done  = 0

    for q in queries:
        print(f"\n[{q['id']}] {q['query'][:70]}")

        for sys_id, kind, model in systems:
            if not reachable.get(sys_id):
                print(f"  {sys_id}: skipped (unreachable)")
                results["systems"][sys_id].append({
                    "id": q["id"], "promql": "",
                    "scores": Score(error="system unreachable").to_dict(),
                    "latency": 0,
                })
                continue

            try:
                if kind == "our":
                    promql, latency = call_our_rag(our_url, q["query"], model, openai_key)
                else:
                    promql, latency = call_prometheus_rag(prag_url, q["query"])

                s = score(
                    promql,
                    q.get("expected_metrics", []),
                    q.get("expected_functions", []),
                    prometheus_url,
                )
                status = "ok" if s.total >= 0.75 else "partial" if s.total > 0 else "fail"
                print(f"  {sys_id}: {promql[:60]!r} | score={s.total:.2f} [{status}]")

            except Exception as e:
                promql, latency = "", 0
                s = Score(error=str(e))
                print(f"  {sys_id}: ERROR - {e}")

            results["systems"][sys_id].append({
                "id":         q["id"],
                "query":      q["query"],
                "category":   q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "promql":     promql,
                "scores":     s.to_dict(),
                "latency":    round(latency, 2),
            })

            done += 1
            if delay > 0 and done < total:
                time.sleep(delay)

    return results


def summarize(results: dict) -> dict:
    summary = {}
    for sys_id, entries in results["systems"].items():
        if not entries:
            continue

        scored = [e for e in entries if not e["scores"]["error"]]

        def avg(key):
            vals = [e["scores"][key] for e in scored]
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        cats   = {}
        diffs  = {}
        for e in scored:
            cats.setdefault(e.get("category", "unknown"), []).append(e["scores"]["total"])
            diffs.setdefault(e.get("difficulty", "unknown"), []).append(e["scores"]["total"])

        summary[sys_id] = {
            "total_score":       avg("total"),
            "valid_promql":      avg("valid_promql"),
            "correct_metrics":   avg("correct_metrics"),
            "correct_functions": avg("correct_functions"),
            "executes":          avg("executes"),
            "avg_latency":       round(sum(e["latency"] for e in entries) / len(entries), 2),
            "by_category":       {c: round(sum(v)/len(v), 3) for c, v in cats.items()},
            "by_difficulty":     {d: round(sum(v)/len(v), 3) for d, v in diffs.items()},
            "n":                 len(scored),
        }
    return summary


def print_table(summary: dict):
    systems = list(summary.keys())
    if not systems:
        print("no results to display")
        return

    metrics = ["total_score", "valid_promql", "correct_metrics",
               "correct_functions", "executes", "avg_latency"]
    col_w  = 18
    header = f"{'metric':<22}" + "".join(f"{s:>{col_w}}" for s in systems)
    sep    = "=" * len(header)

    print(f"\n{sep}\nBENCHMARK RESULTS\n{sep}")
    print(header)
    print("-" * len(header))

    for m in metrics:
        row = f"{m:<22}"
        for s in systems:
            val = summary[s].get(m, 0)
            row += f"{val:>{col_w}.2f}s" if m == "avg_latency" else f"{val:>{col_w}.3f}"
        print(row)

    print("-" * len(header))

    all_cats = sorted({c for s in systems for c in summary[s].get("by_category", {})})
    if all_cats:
        print("\nby category (total score):")
        for cat in all_cats:
            row = f"  {cat:<20}"
            for s in systems:
                row += f"{summary[s].get('by_category', {}).get(cat, 0):>{col_w}.3f}"
            print(row)

    print("\nby difficulty (total score):")
    for diff in ["easy", "medium", "hard"]:
        row = f"  {diff:<20}"
        for s in systems:
            row += f"{summary[s].get('by_difficulty', {}).get(diff, 0):>{col_w}.3f}"
        print(row)

    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--our-url",        default="http://localhost:8082")
    parser.add_argument("--prag-url",       default="http://localhost:8080")
    parser.add_argument("--prometheus-url", default="http://localhost:9090")
    parser.add_argument("--openai-key",     default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--openai-model",   default="gpt-4o-mini")
    parser.add_argument("--queries",        default="queries.json")
    parser.add_argument("--output",         default="results.json")
    parser.add_argument("--delay",          type=float, default=1.0)
    args = parser.parse_args()

    with open(args.queries) as f:
        queries = json.load(f)

    print(f"running {len(queries)} queries")
    print(f"our RAG   : {args.our_url}")
    print(f"prag      : {args.prag_url}")
    print(f"prometheus: {args.prometheus_url}")
    print(f"model     : {args.openai_model}")

    results  = run_benchmark(
        queries=queries,
        our_url=args.our_url,
        prag_url=args.prag_url,
        prometheus_url=args.prometheus_url,
        openai_model=args.openai_model,
        openai_key=args.openai_key,
        delay=args.delay,
    )
    summary  = summarize(results)
    results["summary"] = summary

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {args.output}")

    print_table(summary)


if __name__ == "__main__":
    main()