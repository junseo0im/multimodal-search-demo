from types import SimpleNamespace

from src.search.fusion import ScoredCandidate
from src.search.query_analyzer import QueryPlan, SearchWeights
from src.search.unified_search import unified_search


def test_unified_search_routes_scene_search(monkeypatch):
    monkeypatch.setattr(
        "src.search.unified_search.analyze_query",
        lambda query, optional_video_id=None: QueryPlan(
            intent="scene_search",
            scene_query=query,
            weights=SearchWeights(0.6, 0.4),
        ),
    )

    calls = {}

    def fake_hybrid(*args, **kwargs):
        calls["video_id"] = kwargs.get("video_id")
        calls["alpha"] = kwargs.get("alpha")
        return [ScoredCandidate("a", {"video_id": "short_001"}, hybrid_score=1.0)]

    monkeypatch.setattr("src.search.unified_search.hybrid_search", fake_hybrid)

    result = unified_search(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), "\ub300\ud30c \ub123\ub294 \uc7a5\uba74")
    assert result.plan.intent == "scene_search"
    assert result.scenes[0].point_id == "a"
    assert calls["video_id"] is None
    assert calls["alpha"] == 0.6


def test_unified_search_uses_video_id_filter(monkeypatch):
    monkeypatch.setattr(
        "src.search.unified_search.analyze_query",
        lambda query, optional_video_id=None: QueryPlan(
            intent="scene_search",
            scope="video_id",
            scene_query=query,
            weights=SearchWeights(0.6, 0.4),
        ),
    )

    calls = {}

    def fake_hybrid(*args, **kwargs):
        calls["video_id"] = kwargs.get("video_id")
        return [ScoredCandidate("a", {"video_id": "short_001"}, hybrid_score=1.0)]

    monkeypatch.setattr("src.search.unified_search.hybrid_search", fake_hybrid)

    result = unified_search(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        "\ub300\ud30c \ub123\ub294 \uc7a5\uba74",
        optional_video_id="short_001",
    )
    assert result.scenes
    assert calls["video_id"] == "short_001"


def test_unified_search_routes_compound_query(monkeypatch):
    monkeypatch.setattr(
        "src.search.unified_search.analyze_query",
        lambda query, optional_video_id=None: QueryPlan(
            intent="compound_scene_search",
            scope="video_candidate",
            video_query="\uae40\uce58\ucc0c\uac1c \uc601\uc0c1",
            scene_query="\ub300\ud30c \ub123\ub294 \uc7a5\uba74",
            weights=SearchWeights(0.65, 0.35),
        ),
    )

    monkeypatch.setattr(
        "src.search.unified_search.text_search",
        lambda *args, **kwargs: [
            SimpleNamespace(id="v1", score=0.9, payload={"video_id": "short_001", "recipe_name": "\uae40\uce58\ucc0c\uac1c"})
        ],
    )

    calls = {}

    def fake_hybrid(*args, **kwargs):
        calls["video_id"] = kwargs.get("video_id")
        return [ScoredCandidate("a", {"video_id": kwargs.get("video_id")}, hybrid_score=1.0)]

    monkeypatch.setattr("src.search.unified_search.hybrid_search", fake_hybrid)

    result = unified_search(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), "query")
    assert result.plan.intent == "compound_scene_search"
    assert result.videos[0].video_id == "short_001"
    assert result.scenes[0].payload["video_id"] == "short_001"
    assert calls["video_id"] == "short_001"
