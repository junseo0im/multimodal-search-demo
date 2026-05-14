from types import SimpleNamespace

from src.search.dedup import dedup_adjacent
from src.search.fusion import ScoredCandidate, fuse_results, weights_for_query


def result(point_id, score, payload=None):
    return SimpleNamespace(id=point_id, score=score, payload=payload or {"segment_id": point_id})


def test_fuse_results_unions_and_scores():
    fused = fuse_results(
        [result("a", 0.2), result("b", 0.4)],
        [result("b", 0.1), result("c", 0.9)],
        alpha=0.5,
        beta=0.5,
    )
    ids = {item.point_id for item in fused}
    assert ids == {"a", "b", "c"}
    assert fused[0].hybrid_score >= fused[-1].hybrid_score


def test_weights_for_query_prefers_text_for_action():
    alpha, beta = weights_for_query("대파 넣는 타이밍")
    assert alpha > beta


def test_dedup_adjacent_same_video():
    candidates = [
        ScoredCandidate("a", {"video_id": "short_001", "start_time": 10, "end_time": 15}, hybrid_score=1),
        ScoredCandidate("b", {"video_id": "short_001", "start_time": 16, "end_time": 20}, hybrid_score=0.9),
        ScoredCandidate("c", {"video_id": "short_002", "start_time": 16, "end_time": 20}, hybrid_score=0.8),
    ]
    kept = dedup_adjacent(candidates, top_k=5, overlap_threshold=2)
    assert [item.point_id for item in kept] == ["a", "c"]

