from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimeRange:
    video_id: str
    start_time: float
    end_time: float


def temporal_iou(pred: TimeRange, gold: TimeRange) -> float:
    if pred.video_id != gold.video_id:
        return 0.0
    inter = max(0.0, min(pred.end_time, gold.end_time) - max(pred.start_time, gold.start_time))
    union = max(pred.end_time, gold.end_time) - min(pred.start_time, gold.start_time)
    return inter / union if union > 0 else 0.0


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum((1.0 / rank) if rank else 0.0 for rank in ranks) / len(ranks)

