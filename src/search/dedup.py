from __future__ import annotations

from src.search.fusion import ScoredCandidate


def _times(candidate: ScoredCandidate) -> tuple[str, float, float]:
    p = candidate.payload
    return (
        str(p.get("video_id", "")),
        float(p.get("start_time", p.get("current_time", 0.0)) or 0.0),
        float(p.get("end_time", p.get("current_time", 0.0)) or 0.0),
    )


def dedup_adjacent(
    candidates: list[ScoredCandidate],
    top_k: int,
    overlap_threshold: float = 2.0,
) -> list[ScoredCandidate]:
    kept: list[ScoredCandidate] = []
    for cand in candidates:
        vid, start, end = _times(cand)
        duplicate = False
        for existing in kept:
            e_vid, e_start, e_end = _times(existing)
            if vid != e_vid:
                continue
            overlaps = start <= e_end + overlap_threshold and end >= e_start - overlap_threshold
            if overlaps:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
        if len(kept) >= top_k:
            break
    return kept

