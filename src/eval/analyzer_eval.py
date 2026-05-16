from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.eval.run_eval import load_eval_queries
from src.search.query_analyzer import analyze_query


def expected_result_type(expected_intent: str) -> str:
    mapping = {
        "video_search": "video",
        "scene_search": "scene",
        "in_video_search": "in_video",
        "compound_scene_search": "compound",
        "summary": "summary",
    }
    return mapping.get(expected_intent, "")


def evaluate_analyzer(queries: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for _, row in queries.iterrows():
        query = str(row["query"])
        target_video_id = str(row.get("target_video_id", "") or "").strip() or None
        plan = analyze_query(query, target_video_id)
        expected_intent_value = str(row.get("expected_intent", ""))
        expected_result_value = str(row.get("expected_result_type", "")) or expected_result_type(expected_intent_value)
        expected_scope_value = str(row.get("expected_scope", ""))
        rows.append(
            {
                "query": query,
                "expected_intent": expected_intent_value,
                "predicted_intent": plan.intent,
                "intent_correct": expected_intent_value == plan.intent,
                "expected_result_type": expected_result_value,
                "predicted_result_type": expected_result_type(plan.intent),
                "result_type_correct": expected_result_value == expected_result_type(plan.intent),
                "expected_scope": expected_scope_value,
                "predicted_scope": plan.scope,
                "scope_correct": (not expected_scope_value) or expected_scope_value == plan.scope,
                "video_query": plan.video_query,
                "scene_query": plan.scene_query,
                "weights_text": plan.weights.text,
                "weights_image": plan.weights.image,
                "analyzer": plan.analyzer,
                "fallback_reason": plan.fallback_reason,
            }
        )
    detail = pd.DataFrame(rows)
    total = len(detail)
    summary = {
        "queries": float(total),
        "intent_accuracy": float(detail["intent_correct"].mean()) if total else 0.0,
        "result_type_accuracy": float(detail["result_type_correct"].mean()) if total else 0.0,
        "scope_accuracy": float(detail["scope_correct"].mean()) if total else 0.0,
    }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Query Analyzer routing")
    parser.add_argument("--queries", required=True, help="Path to evaluation query CSV or Parquet")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    queries = load_eval_queries(args.queries)
    detail, summary = evaluate_analyzer(queries)
    print(pd.DataFrame([summary]).to_string(index=False))
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
