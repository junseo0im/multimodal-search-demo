from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from src.eval.metrics import TimeRange, mean_reciprocal_rank, recall_at_k, temporal_iou
from src.index.build_index import COLLECTION_NAME
from src.index.qdrant_client import get_qdrant_client
from src.models.bge_encoder import BGEEncoder
from src.models.siglip_encoder import SigLIPEncoder
from src.search.hybrid_search import hybrid_search, image_search, text_search


def _load_positive_segments(value: Any) -> list[TimeRange]:
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


def _candidate_range(candidate: Any) -> TimeRange:
    payload = dict(candidate.payload or {})
    return TimeRange(
        video_id=str(payload.get("video_id", "")),
        start_time=float(payload.get("start_time", payload.get("current_time", 0.0)) or 0.0),
        end_time=float(payload.get("end_time", payload.get("current_time", 0.0)) or 0.0),
    )


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


def evaluate_queries(
    queries: pd.DataFrame,
    client: QdrantClient,
    bge: BGEEncoder,
    siglip: SigLIPEncoder,
    mode: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    iou_threshold: float = 0.3,
) -> dict[str, float]:
    ranks: list[int | None] = []
    best_ious: list[float] = []
    for _, row in queries.iterrows():
        query = str(row["query"])
        positives = _load_positive_segments(row["positive_segments"])
        video_id = str(row.get("target_video_id", "") or "").strip() or None
        if mode == "text-only":
            results = text_search(client, bge, query, collection_name, top_k, video_id)
        elif mode == "image-only":
            results = image_search(client, siglip, query, collection_name, top_k, video_id)
        elif mode == "hybrid":
            results = hybrid_search(client, bge, siglip, query, collection_name, top_k=top_k, video_id=video_id)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        rank, best_iou = _rank_for_results(results, positives, iou_threshold)
        ranks.append(rank)
        best_ious.append(best_iou)
    return {
        "mode": mode,
        "queries": float(len(queries)),
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "mrr": mean_reciprocal_rank(ranks),
        "mean_best_iou": sum(best_ious) / len(best_ious) if best_ious else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MVP search pipeline")
    parser.add_argument("--queries", required=True, help="Path to evaluation_queries.parquet")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    queries = pd.read_parquet(Path(args.queries))
    client = get_qdrant_client()
    bge = BGEEncoder()
    siglip = SigLIPEncoder(adapter_path=args.adapter_path)
    rows = [
        evaluate_queries(queries, client, bge, siglip, mode, args.collection, args.top_k)
        for mode in ("text-only", "image-only", "hybrid")
    ]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

