from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.metrics import TimeRange, mean_reciprocal_rank, rank_for_video_ids, recall_at_k, temporal_iou


COLLECTION_NAME = "cooking_segments"


REQUIRED_QUERY_COLUMNS = [
    "query",
    "query_type",
    "expected_intent",
    "expected_result_type",
    "positive_segments",
]

OPTIONAL_QUERY_COLUMNS = {
    "positive_video_ids": "",
    "target_video_id": "",
    "notes": "",
}


def load_eval_queries(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        queries = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    elif path.suffix.lower() in {".parquet", ".pq"}:
        queries = pd.read_parquet(path).fillna("")
    else:
        raise ValueError(f"Unsupported query file format: {path.suffix}")
    for column, default in OPTIONAL_QUERY_COLUMNS.items():
        if column not in queries.columns:
            queries[column] = default
    validate_eval_queries(queries)
    return queries


def validate_eval_queries(queries: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_QUERY_COLUMNS if col not in queries.columns]
    if missing:
        raise ValueError(f"Evaluation queries missing columns: {missing}")
    for idx, value in enumerate(queries["positive_segments"].tolist(), start=1):
        segments = _load_positive_segments(value)
        positive_video_ids = _load_positive_video_ids(queries.iloc[idx - 1].get("positive_video_ids", ""), segments)
        result_type = str(queries.iloc[idx - 1].get("expected_result_type", ""))
        if not segments and result_type != "video":
            raise ValueError(f"Row {idx} has no positive segments")
        if result_type == "video" and not positive_video_ids:
            raise ValueError(f"Row {idx} has no positive video ids")


def _load_positive_segments(value: Any) -> list[TimeRange]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return [
        TimeRange(
            video_id=str(item["video_id"]),
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
        )
        for item in parsed
    ]


def _load_positive_video_ids(value: Any, fallback_segments: list[TimeRange] | None = None) -> set[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            parsed: Any = []
        elif stripped.startswith("["):
            parsed = json.loads(stripped)
        else:
            parsed = [part.strip() for part in stripped.replace(";", ",").split(",")]
    elif value is None:
        parsed = []
    else:
        parsed = value
    video_ids = {str(item).strip() for item in parsed if str(item).strip()}
    if not video_ids and fallback_segments:
        video_ids = {segment.video_id for segment in fallback_segments if segment.video_id}
    return video_ids


def _candidate_range(candidate: Any) -> TimeRange:
    payload = dict(candidate.payload or {})
    return TimeRange(
        video_id=str(payload.get("video_id", "")),
        start_time=float(payload.get("start_time", payload.get("current_time", 0.0)) or 0.0),
        end_time=float(payload.get("end_time", payload.get("current_time", 0.0)) or 0.0),
    )


def _candidate_video_id(candidate: Any) -> str:
    video_id = getattr(candidate, "video_id", "")
    if video_id:
        return str(video_id)
    payload = dict(getattr(candidate, "payload", None) or {})
    return str(payload.get("video_id", ""))


def _rank_for_results(results: list[Any], positives: list[TimeRange], iou_threshold: float) -> tuple[int | None, float]:
    best_iou = 0.0
    for idx, result in enumerate(results, start=1):
        pred = _candidate_range(result)
        ious = [temporal_iou(pred, gold) for gold in positives]
        local_best = max(ious) if ious else 0.0
        best_iou = max(best_iou, local_best)
        if local_best >= iou_threshold:
            return idx, best_iou
    return None, best_iou


def _rank_for_video_results(results: list[Any], positive_video_ids: set[str]) -> int | None:
    return rank_for_video_ids([_candidate_video_id(result) for result in results], positive_video_ids)


def _is_video_eval(row: pd.Series) -> bool:
    expected_result_type = str(row.get("expected_result_type", "")).strip()
    expected_intent = str(row.get("expected_intent", "")).strip()
    query_type = str(row.get("query_type", "")).strip()
    return expected_result_type == "video" or expected_intent == "video_search" or query_type == "recipe"


def _search_results_for_mode(
    row: pd.Series,
    client: Any,
    bge: Any,
    siglip: Any,
    mode: str,
    collection_name: str,
    top_k: int,
) -> list[Any]:
    from src.search.hybrid_search import hybrid_search, image_search, text_search
    from src.search.unified_search import unified_search

    query = str(row["query"])
    video_id = str(row.get("target_video_id", "") or "").strip() or None
    if mode == "text-only":
        return text_search(client, bge, query, collection_name, top_n=max(top_k * 4, top_k), video_id=video_id)
    if mode == "image-only":
        return image_search(client, siglip, query, collection_name, top_n=max(top_k * 4, top_k), video_id=video_id)
    if mode == "hybrid":
        return hybrid_search(client, bge, siglip, query, collection_name, top_k=max(top_k * 4, top_k), video_id=video_id)
    if mode == "unified":
        unified = unified_search(
            client,
            bge,
            siglip,
            query,
            collection_name=collection_name,
            top_k=top_k,
            optional_video_id=video_id,
        )
        return unified.videos if _is_video_eval(row) else unified.scenes
    raise ValueError(f"Unknown mode: {mode}")


def evaluate_queries(
    queries: pd.DataFrame,
    client: Any,
    bge: Any,
    siglip: Any,
    mode: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    iou_threshold: float = 0.3,
) -> dict[str, float]:
    ranks: list[int | None] = []
    best_ious: list[float] = []
    for _, row in queries.iterrows():
        positives = _load_positive_segments(row["positive_segments"])
        results = _search_results_for_mode(row, client, bge, siglip, mode, collection_name, top_k)
        if _is_video_eval(row):
            positive_video_ids = _load_positive_video_ids(row.get("positive_video_ids", ""), positives)
            rank = _rank_for_video_results(results, positive_video_ids)
            best_iou = 0.0
        else:
            rank, best_iou = _rank_for_results(results, positives, iou_threshold)
        ranks.append(rank)
        best_ious.append(best_iou)
    return {
        "mode": mode,
        "query_type": "overall",
        "queries": float(len(queries)),
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "mrr": mean_reciprocal_rank(ranks),
        "mean_best_iou": sum(best_ious) / len(best_ious) if best_ious else 0.0,
    }


def evaluate_queries_by_type(
    queries: pd.DataFrame,
    client: Any,
    bge: Any,
    siglip: Any,
    mode: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    iou_threshold: float = 0.3,
) -> list[dict[str, float]]:
    rows = [evaluate_queries(queries, client, bge, siglip, mode, collection_name, top_k, iou_threshold)]
    for query_type, group in queries.groupby("query_type", sort=True):
        metrics = evaluate_queries(group, client, bge, siglip, mode, collection_name, top_k, iou_threshold)
        metrics["query_type"] = str(query_type)
        rows.append(metrics)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search pipeline")
    parser.add_argument("--queries", required=True, help="Path to evaluation query CSV or Parquet")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=["text-only", "image-only", "hybrid", "unified"])
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    from src.index.qdrant_client import get_qdrant_client
    from src.models.bge_encoder import BGEEncoder
    from src.models.siglip_encoder import SigLIPEncoder

    queries = load_eval_queries(Path(args.queries))
    client = get_qdrant_client()
    bge = BGEEncoder()
    siglip = SigLIPEncoder(adapter_path=args.adapter_path)
    rows = []
    for mode in args.modes:
        rows.extend(evaluate_queries_by_type(queries, client, bge, siglip, mode, args.collection, args.top_k))
    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
