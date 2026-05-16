import pandas as pd

from src.eval.analyzer_eval import evaluate_analyzer, expected_result_type
from src.eval.run_eval import load_eval_queries, validate_eval_queries


def test_eval_query_template_loads():
    queries = load_eval_queries("templates/evaluation_queries.csv")
    assert len(queries) >= 10
    assert {"query", "positive_segments", "expected_intent", "expected_result_type"} <= set(queries.columns)


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
