import pandas as pd
from types import SimpleNamespace

from src.eval.analyzer_eval import evaluate_analyzer, expected_result_type
from src.eval import run_eval
from src.eval.run_eval import evaluate_queries, evaluate_queries_by_type, load_eval_queries, validate_eval_queries


def test_eval_query_template_loads():
    queries = load_eval_queries("templates/evaluation_queries.csv")
    assert len(queries) >= 10
    assert {"query", "positive_segments", "expected_intent", "expected_result_type"} <= set(queries.columns)


def test_retrieval_eval_template_has_expected_columns():
    queries = pd.read_csv("templates/retrieval_eval_queries.csv")
    assert {
        "query",
        "query_type",
        "expected_intent",
        "expected_result_type",
        "positive_segments",
        "positive_video_ids",
        "target_video_id",
        "notes",
    } <= set(queries.columns)


def test_validate_eval_queries_rejects_missing_columns():
    df = pd.DataFrame({"query": ["x"]})
    try:
        validate_eval_queries(df)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("Expected missing columns validation error")


def test_expected_result_type_mapping():
    assert expected_result_type("video_search") == "video"
    assert expected_result_type("compound_scene_search") == "compound"
    assert expected_result_type("summary") == "summary"


def test_evaluate_analyzer_summary(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    df = pd.DataFrame(
        [
            {
                "query": "\uc774 \uc601\uc0c1 \uc7ac\ub8cc \uc815\ub9ac\ud574\uc918",
                "query_type": "summary",
                "expected_intent": "summary",
                "expected_result_type": "summary",
                "expected_scope": "",
                "positive_segments": '[{"video_id": "short_001", "start_time": 0, "end_time": 1}]',
                "target_video_id": "short_001",
                "notes": "",
            }
        ]
    )
    detail, summary = evaluate_analyzer(df)
    assert detail.iloc[0]["predicted_intent"] == "summary"
    assert summary["intent_accuracy"] == 1.0


def test_video_search_eval_uses_positive_video_ids(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "query": "김치찌개 영상 찾아줘",
                "query_type": "recipe",
                "expected_intent": "video_search",
                "expected_result_type": "video",
                "positive_segments": "[]",
                "positive_video_ids": '["short_002"]',
                "target_video_id": "",
                "notes": "",
            }
        ]
    )

    monkeypatch.setattr(
        run_eval,
        "_search_results_for_mode",
        lambda *args, **kwargs: [
            SimpleNamespace(payload={"video_id": "short_001"}, score=0.9),
            SimpleNamespace(payload={"video_id": "short_002"}, score=0.8),
        ],
    )

    metrics = evaluate_queries(df, None, None, None, "text-only")

    assert metrics["recall@1"] == 0.0
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr"] == 0.5


def test_evaluate_queries_by_type_adds_group_rows(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "query": "대파 넣는 장면",
                "query_type": "ingredient_action",
                "expected_intent": "scene_search",
                "expected_result_type": "scene",
                "positive_segments": '[{"video_id": "short_001", "start_time": 10, "end_time": 12}]',
                "positive_video_ids": "",
                "target_video_id": "",
                "notes": "",
            }
        ]
    )

    monkeypatch.setattr(
        run_eval,
        "_search_results_for_mode",
        lambda *args, **kwargs: [
            SimpleNamespace(payload={"video_id": "short_001", "start_time": 10, "end_time": 12}, score=0.9)
        ],
    )

    rows = evaluate_queries_by_type(df, None, None, None, "text-only")

    assert [row["query_type"] for row in rows] == ["overall", "ingredient_action"]
    assert rows[1]["recall@1"] == 1.0
