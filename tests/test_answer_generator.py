from src.generation.answer_generator import generate_answer
from src.search.fusion import ScoredCandidate
from src.search.query_analyzer import QueryPlan
from src.search.unified_search import UnifiedSearchResult


def test_generate_answer_returns_empty_for_non_summary():
    result = UnifiedSearchResult(plan=QueryPlan(intent="scene_search"), scenes=[], videos=[], result_type="scene")
    assert generate_answer("query", result) == ""


def test_generate_answer_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = UnifiedSearchResult(
        plan=QueryPlan(intent="summary", needs_generation=True),
        scenes=[
            ScoredCandidate(
                "a",
                {
                    "recipe_name": "\uae40\uce58\ucc0c\uac1c",
                    "video_id": "short_001",
                    "start_time": 1.0,
                    "end_time": 3.0,
                    "caption": "\ub300\ud30c\ub97c \ub123\ub294\ub2e4",
                },
                hybrid_score=1.0,
            )
        ],
        videos=[],
        answer_context="context",
        result_type="summary",
    )
    answer = generate_answer("\uc774 \uc601\uc0c1 \uc7ac\ub8cc \uc815\ub9ac\ud574\uc918", result)
    assert "GEMINI_API_KEY" in answer
    assert "short_001" in answer


def test_generate_answer_handles_empty_scenes():
    result = UnifiedSearchResult(plan=QueryPlan(intent="summary"), scenes=[], videos=[], result_type="summary")
    assert "\ub2f5\ubcc0\uc744 \uc0dd\uc131\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4" in generate_answer("query", result)
