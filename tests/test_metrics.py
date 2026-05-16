from src.eval.metrics import TimeRange, mean_reciprocal_rank, rank_for_video_ids, recall_at_k, temporal_iou


def test_temporal_iou_same_video():
    pred = TimeRange("short_001", 10, 20)
    gold = TimeRange("short_001", 15, 25)
    assert round(temporal_iou(pred, gold), 4) == 0.3333


def test_temporal_iou_different_video_is_zero():
    assert temporal_iou(TimeRange("a", 0, 1), TimeRange("b", 0, 1)) == 0.0


def test_rank_metrics():
    ranks = [1, 5, None]
    assert recall_at_k(ranks, 1) == 1 / 3
    assert recall_at_k(ranks, 5) == 2 / 3
    assert mean_reciprocal_rank(ranks) == (1 + 0.2) / 3


def test_rank_for_video_ids_deduplicates_candidates():
    rank = rank_for_video_ids(["short_001", "short_001", "short_003"], {"short_003"})
    assert rank == 2
