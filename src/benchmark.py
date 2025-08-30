"""Benchmark latency before vs after optimizations.

Modes simulated:
  baseline  - Disables new optimizations: no parallel compare, clears vectorstore cache each run.
  optimized - Uses current CONFIG settings (caching, parallel, etc.).

By default measures retrieval-only (no answer generation) to avoid extra token costs.
Optionally measures full answer / compare if --full is passed (costly due to LLM calls).
"""
from __future__ import annotations
import argparse
import time
from statistics import mean, stdev
from typing import List

from .config import CONFIG
from . import qa
from .retriever import retrieve

try:  # access cached store for clearing
    from .retriever import _cached_store  # type: ignore
except Exception:  # pragma: no cover
    _cached_store = None  # type: ignore


def clear_vector_cache():
    if _cached_store and hasattr(_cached_store, 'cache_clear'):
        _cached_store.cache_clear()


def run_retrieval(question: str, docs: List[str]):
    results = []
    start = time.perf_counter()
    for d in docs:
        _ = retrieve(question, d)
    elapsed = time.perf_counter() - start
    results.append(elapsed)
    return results


def run_full(question: str, docs: List[str]):
    start = time.perf_counter()
    if len(docs) == 1:
        qa.answer(question, docs[0])
    else:
        qa.compare(question, docs)
    return [time.perf_counter() - start]


def benchmark(question: str, docs: List[str], runs: int, full: bool, fast: bool):
    modes = ["baseline", "optimized"]
    report = {}
    # Snapshot current optimized settings
    opt_parallel = CONFIG.parallel_compare
    opt_disable_rerank = CONFIG.disable_rerank

    for mode in modes:
        times = []
        for i in range(runs):
            # Configure per mode
            if mode == "baseline":
                CONFIG.parallel_compare = False
                CONFIG.disable_rerank = False  # keep LLM rerank
                clear_vector_cache()
            else:  # optimized
                CONFIG.parallel_compare = opt_parallel
                CONFIG.disable_rerank = opt_disable_rerank
                # keep cache warm
                if fast:
                    # Apply fast profile (temporary)
                    CONFIG.disable_rerank = True
                    CONFIG.top_k_initial = min(15, CONFIG.top_k_initial)
                    CONFIG.top_k_pages = min(5, CONFIG.top_k_pages)

            if full:
                times.extend(run_full(question, docs))
            else:
                times.extend(run_retrieval(question, docs))

        report[mode] = {
            "runs": runs,
            "avg_s": round(mean(times), 3),
            "stdev_s": round(stdev(times), 3) if len(times) > 1 else 0.0,
            "samples": [round(t, 3) for t in times]
        }

    # Restore settings
    CONFIG.parallel_compare = opt_parallel
    CONFIG.disable_rerank = opt_disable_rerank
    return report


def main():
    ap = argparse.ArgumentParser(description="Latency benchmark for retrieval/QA")
    ap.add_argument("question")
    ap.add_argument("docs", nargs='+', help="One or more document stems")
    ap.add_argument("--runs", type=int, default=3, help="Number of runs per mode")
    ap.add_argument("--full", action="store_true", help="Measure full answer/compare (costly)")
    ap.add_argument("--fast", action="store_true", help="Apply fast profile to optimized mode (skip rerank, smaller K)")
    args = ap.parse_args()

    rep = benchmark(args.question, args.docs, args.runs, args.full, args.fast)
    kind = "FULL QA" if args.full else "RETRIEVE ONLY"
    print(f"Benchmark kind: {kind}\nQuestion: {args.question}\nDocs: {', '.join(args.docs)}")
    for mode, stats in rep.items():
        print(f"\nMode: {mode}\n  runs: {stats['runs']}\n  avg_s: {stats['avg_s']}\n  stdev_s: {stats['stdev_s']}\n  samples_s: {stats['samples']}")
    speedup = rep['baseline']['avg_s'] / rep['optimized']['avg_s'] if rep['optimized']['avg_s'] else None
    if speedup:
        print(f"\nEstimated speedup (baseline/optimized): {round(speedup, 2)}x")


if __name__ == "__main__":
    main()
