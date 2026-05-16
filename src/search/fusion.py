from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.search.query_analyzer import weights_for_query as analyze_weights


@dataclass
class ScoredCandidate:
    point_id: str
    payload: dict[str, Any]
    text_score: float = 0.0
    image_score: float = 0.0
    hybrid_score: float = 0.0


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def fuse_results(
    text_results: list[Any],
    image_results: list[Any],
    alpha: float = 0.6,
    beta: float = 0.4,
) -> list[ScoredCandidate]:
    candidates: dict[str, ScoredCandidate] = {}

    text_norm = minmax([float(r.score) for r in text_results])
    image_norm = minmax([float(r.score) for r in image_results])

    for result, score in zip(text_results, text_norm):
        point_id = str(result.id)
        candidates[point_id] = ScoredCandidate(
            point_id=point_id,
            payload=dict(result.payload or {}),
            text_score=score,
        )

    for result, score in zip(image_results, image_norm):
        point_id = str(result.id)
        if point_id not in candidates:
            candidates[point_id] = ScoredCandidate(
                point_id=point_id,
                payload=dict(result.payload or {}),
            )
        candidates[point_id].image_score = score

    for candidate in candidates.values():
        candidate.hybrid_score = alpha * candidate.text_score + beta * candidate.image_score

    return sorted(candidates.values(), key=lambda c: c.hybrid_score, reverse=True)


def weights_for_query(query: str) -> tuple[float, float]:
    weights = analyze_weights(query)
    return weights.text, weights.image
