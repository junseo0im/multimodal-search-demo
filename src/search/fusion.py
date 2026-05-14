from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    visual_terms = ("보이는", "장면", "색", "모양", "완성", "비주얼", "프라이팬", "그릇")
    text_terms = ("넣", "볶", "끓", "섞", "자르", "타이밍", "몇 분", "언제", "재료")
    visual_hits = sum(term in query for term in visual_terms)
    text_hits = sum(term in query for term in text_terms)
    if visual_hits > text_hits:
        return 0.4, 0.6
    if text_hits > visual_hits:
        return 0.7, 0.3
    return 0.6, 0.4

