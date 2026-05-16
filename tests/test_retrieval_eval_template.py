from pathlib import Path

import pandas as pd

from src.eval.build_retrieval_eval_template import build_query_specs_from_segments, build_retrieval_eval_template, load_segments


def test_build_retrieval_eval_template_matches_scene_keywords():
    segments = pd.DataFrame(
        [
            {
                "video_id": "short_001",
                "recipe_name": "김치찌개",
                "caption": "대파를 넣습니다",
                "start_time": 10.0,
                "end_time": 12.0,
            },
            {
                "video_id": "short_002",
                "recipe_name": "떡볶이",
                "caption": "고추장을 넣습니다",
                "start_time": 3.0,
                "end_time": 5.0,
            },
        ]
    )
    specs = pd.DataFrame(
        [
            {
                "query": "김치찌개 영상에서 대파 넣는 장면",
                "query_type": "compound",
                "expected_intent": "compound_scene_search",
                "expected_result_type": "compound",
                "recipe_keywords": "김치찌개",
                "scene_keywords": "대파+넣",
                "target_video_id": "",
                "notes": "",
            }
        ]
    )

    result = build_retrieval_eval_template(segments, specs)

    assert result.iloc[0]["auto_match_count"] == 1
    assert "short_001" in result.iloc[0]["positive_segments"]
    assert bool(result.iloc[0]["needs_review"]) is True


def test_visual_state_query_requires_review():
    segments = pd.DataFrame(
        [
            {
                "video_id": "short_001",
                "recipe_name": "계란 요리",
                "caption": "계란이 익은 장면",
                "start_time": 1.0,
                "end_time": 2.0,
            }
        ]
    )
    specs = pd.DataFrame(
        [
            {
                "query": "계란이 익은 장면",
                "query_type": "visual_state",
                "expected_intent": "scene_search",
                "expected_result_type": "scene",
                "recipe_keywords": "",
                "scene_keywords": "익",
                "target_video_id": "",
                "notes": "",
            }
        ]
    )

    result = build_retrieval_eval_template(segments, specs)

    assert bool(result.iloc[0]["needs_review"]) is True
    assert "Visual-state query" in result.iloc[0]["notes"]


def test_build_query_specs_from_segments_uses_actual_recipes_and_captions():
    segments = pd.DataFrame(
        [
            {
                "video_id": "short_001",
                "recipe_name": "김치찌개",
                "caption": "대파를 넣고 보글보글 끓입니다",
                "start_time": 1.0,
                "end_time": 2.0,
            },
            {
                "video_id": "short_002",
                "recipe_name": "떡볶이",
                "caption": "고추장을 넣고 섞습니다",
                "start_time": 3.0,
                "end_time": 4.0,
            },
        ]
    )

    specs = build_query_specs_from_segments(
        segments,
        recipe_count=2,
        ingredient_action_count=2,
        visual_state_count=1,
        compound_count=2,
    )

    assert "김치찌개 영상 찾아줘" in specs["query"].tolist()
    assert any("대파" in query for query in specs["query"].tolist())
    assert set(specs["query_type"]) >= {"recipe", "ingredient_action", "compound"}


def test_load_segments_joins_recipe_name_from_urls(tmp_path: Path):
    json_path = tmp_path / "segments.json"
    urls_path = tmp_path / "urls.csv"
    json_path.write_text(
        '[{"video_id": "short_001", "caption": "대파를 넣습니다", "start_time": 1, "end_time": 2}]',
        encoding="utf-8",
    )
    urls_path.write_text("video_id,url,memo\nshort_001,https://example.com,김치찌개 요리\n", encoding="utf-8-sig")

    segments = load_segments(json_path, urls_path)

    assert segments.iloc[0]["recipe_name"] == "김치찌개"
    assert segments.iloc[0]["youtube_url"] == "https://example.com"
