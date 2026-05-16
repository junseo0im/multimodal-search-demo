from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SPEC_COLUMNS = [
    "query",
    "query_type",
    "expected_intent",
    "expected_result_type",
    "recipe_keywords",
    "scene_keywords",
    "target_video_id",
    "notes",
]

OUTPUT_COLUMNS = [
    "query",
    "query_type",
    "expected_intent",
    "expected_result_type",
    "positive_segments",
    "positive_video_ids",
    "target_video_id",
    "auto_match_count",
    "needs_review",
    "notes",
]

TEXT_FIELDS = ["recipe_name", "title_text", "asr_text", "ocr_text", "scene_caption", "caption"]
RECIPE_TEXT_FIELDS = ["recipe_name", "title_text", "caption"]
SCENE_TEXT_FIELDS = ["asr_text", "ocr_text", "scene_caption", "caption"]
INGREDIENT_KEYWORDS = [
    "대파",
    "고추장",
    "된장",
    "간장",
    "마늘",
    "양파",
    "참기름",
    "설탕",
    "고춧가루",
    "두부",
    "계란",
    "치즈",
    "소금",
    "후추",
    "김치",
]
ACTION_KEYWORDS = ["넣", "붓", "볶", "끓", "굽", "섞", "버무리", "썰", "자르", "올리", "담"]
VISUAL_STATE_KEYWORDS = ["완성", "노릇", "빨갛", "보글", "녹", "바삭", "익", "졸", "담긴", "그릇", "끓", "볶", "굽"]
ACTION_QUERY_LABELS = {
    "넣": "재료를 넣는 장면",
    "붓": "붓는 장면",
    "볶": "볶는 장면",
    "끓": "끓이는 장면",
    "굽": "굽는 장면",
    "섞": "섞는 장면",
    "버무리": "버무리는 장면",
    "썰": "써는 장면",
    "자르": "자르는 장면",
    "올리": "올리는 장면",
    "담": "담는 장면",
}
VISUAL_QUERY_LABELS = {
    "완성": "완성된 장면",
    "노릇": "노릇하게 익은 장면",
    "빨갛": "빨갛게 양념된 장면",
    "보글": "보글보글 끓는 장면",
    "녹": "치즈나 재료가 녹은 장면",
    "바삭": "바삭하게 조리된 장면",
    "익": "재료가 익은 장면",
    "졸": "국물이 졸아든 장면",
    "담긴": "그릇에 담긴 장면",
    "그릇": "그릇에 담긴 완성 장면",
    "끓": "끓고 있는 장면",
    "볶": "볶아진 장면",
    "굽": "구워진 장면",
}


def load_query_specs(path: str | Path) -> pd.DataFrame:
    specs = pd.read_csv(path, encoding="utf-8-sig").fillna("")
    missing = [column for column in SPEC_COLUMNS if column not in specs.columns]
    if missing:
        raise ValueError(f"Query spec missing columns: {missing}")
    return specs


def _split_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    normalized = text.replace(";", ",").replace("|", ",")
    return [item.strip().lower() for item in normalized.split(",") if item.strip()]


def _row_text(row: pd.Series, fields: list[str] = TEXT_FIELDS) -> str:
    parts = []
    for field in fields:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip():
            parts.append(str(value))
    return " ".join(parts).lower()


def _keyword_mask(segments: pd.DataFrame, keywords: list[str], fields: list[str] = TEXT_FIELDS) -> pd.Series:
    if not keywords:
        return pd.Series([True] * len(segments), index=segments.index)
    texts = segments.apply(lambda row: _row_text(row, fields), axis=1)
    return texts.apply(
        lambda text: any(
            all(part.strip() in text for part in keyword.split("+") if part.strip())
            for keyword in keywords
        )
    )


def _contains_keyword_count(segments: pd.DataFrame, keyword: str) -> int:
    if not keyword:
        return 0
    texts = segments.apply(lambda row: _row_text(row), axis=1)
    return int(texts.str.contains(keyword, regex=False).sum())


def _available_keywords(segments: pd.DataFrame, keywords: list[str]) -> list[tuple[str, int]]:
    counts = [(keyword, _contains_keyword_count(segments, keyword)) for keyword in keywords]
    return sorted([item for item in counts if item[1] > 0], key=lambda item: item[1], reverse=True)


def _top_recipes(segments: pd.DataFrame, limit: int) -> list[tuple[str, int]]:
    if "recipe_name" not in segments.columns:
        return []
    counts = (
        segments["recipe_name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )
    return [(str(recipe), int(count)) for recipe, count in counts.head(limit).items()]


def _recipe_mask(segments: pd.DataFrame, recipe: str) -> pd.Series:
    if "recipe_name" not in segments.columns:
        return pd.Series([False] * len(segments), index=segments.index)
    return segments["recipe_name"].fillna("").astype(str).str.strip() == recipe


def _scene_query_for_keyword(keyword: str) -> str:
    ingredient = keyword.split("+", 1)[0]
    if ingredient in INGREDIENT_KEYWORDS:
        return f"{ingredient} 넣는 장면"
    if keyword in ACTION_QUERY_LABELS:
        return ACTION_QUERY_LABELS[keyword]
    return f"{keyword} 장면"


def build_query_specs_from_segments(
    segments: pd.DataFrame,
    recipe_count: int = 8,
    ingredient_action_count: int = 14,
    visual_state_count: int = 10,
    compound_count: int = 8,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    recipes = _top_recipes(segments, recipe_count)
    for recipe, _ in recipes:
        rows.append(
            {
                "query": f"{recipe} 영상 찾아줘",
                "query_type": "recipe",
                "expected_intent": "video_search",
                "expected_result_type": "video",
                "recipe_keywords": recipe,
                "scene_keywords": "",
                "target_video_id": "",
                "notes": "데이터셋 recipe_name 기반 자동 생성",
            }
        )

    ingredient_keywords = _available_keywords(segments, INGREDIENT_KEYWORDS)
    action_keywords = _available_keywords(segments, ACTION_KEYWORDS)
    scene_queries: list[tuple[str, str]] = []
    ingredient_limit = max(0, ingredient_action_count - 4)
    for ingredient, _ in ingredient_keywords[:ingredient_limit]:
        if len(scene_queries) >= ingredient_action_count:
            break
        scene_queries.append((f"{ingredient} 넣는 장면", f"{ingredient}+넣"))
    for action, _ in action_keywords:
        if len(scene_queries) >= ingredient_action_count:
            break
        scene_queries.append((ACTION_QUERY_LABELS.get(action, f"{action}는 장면"), action))
    for query, keyword in scene_queries[:ingredient_action_count]:
        rows.append(
            {
                "query": query,
                "query_type": "ingredient_action",
                "expected_intent": "scene_search",
                "expected_result_type": "scene",
                "recipe_keywords": "",
                "scene_keywords": keyword,
                "target_video_id": "",
                "notes": "caption/metadata 키워드 기반 자동 생성",
            }
        )

    for keyword, _ in _available_keywords(segments, VISUAL_STATE_KEYWORDS)[:visual_state_count]:
        rows.append(
            {
                "query": VISUAL_QUERY_LABELS.get(keyword, f"{keyword} 장면"),
                "query_type": "visual_state",
                "expected_intent": "scene_search",
                "expected_result_type": "scene",
                "recipe_keywords": "",
                "scene_keywords": keyword,
                "target_video_id": "",
                "notes": "시각 상태 키워드 기반 자동 생성. 프레임 검수 필요",
            }
        )

    compound_rows = []
    for recipe, _ in recipes:
        recipe_segments = segments[_recipe_mask(segments, recipe)]
        for keyword, _ in ingredient_keywords + action_keywords:
            scene_keyword = f"{keyword}+넣" if keyword in INGREDIENT_KEYWORDS else keyword
            if _keyword_mask(recipe_segments, [scene_keyword], SCENE_TEXT_FIELDS).any():
                compound_rows.append((recipe, scene_keyword))
                break
        if len(compound_rows) >= compound_count:
            break
    for recipe, keyword in compound_rows:
        rows.append(
            {
                "query": f"{recipe} 영상에서 {_scene_query_for_keyword(keyword)}",
                "query_type": "compound",
                "expected_intent": "compound_scene_search",
                "expected_result_type": "compound",
                "recipe_keywords": recipe,
                "scene_keywords": keyword,
                "target_video_id": "",
                "notes": "동일 recipe 내 실제 caption 키워드 기반 자동 생성",
            }
        )

    return pd.DataFrame(rows, columns=SPEC_COLUMNS)


def _clean_recipe_memo(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s*요리\s*$", "", text)
    return text.strip()


def load_segments(path: str | Path, urls_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        segments = pd.read_parquet(path).fillna("")
    elif path.suffix.lower() == ".json":
        segments = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))).fillna("")
    else:
        raise ValueError(f"Unsupported segment file format: {path.suffix}")

    if urls_path:
        urls = pd.read_csv(urls_path, encoding="utf-8-sig").fillna("")
        if "memo" in urls.columns and "recipe_name" not in urls.columns:
            urls["recipe_name"] = urls["memo"].apply(_clean_recipe_memo)
        url_columns = [column for column in ["video_id", "recipe_name", "url"] if column in urls.columns]
        if "video_id" in url_columns:
            segments = segments.merge(urls[url_columns], on="video_id", how="left")
            if "url" in segments.columns and "youtube_url" not in segments.columns:
                segments = segments.rename(columns={"url": "youtube_url"})

    if "recipe_name" not in segments.columns:
        segments["recipe_name"] = ""
    return segments.fillna("")


def _segment_payloads(rows: pd.DataFrame, max_positives: int) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for _, row in rows.head(max_positives).iterrows():
        start_time = float(row.get("start_time", row.get("current_time", 0.0)) or 0.0)
        end_time = float(row.get("end_time", row.get("current_time", start_time)) or start_time)
        if end_time <= start_time:
            end_time = start_time + 1.0
        payloads.append(
            {
                "video_id": str(row.get("video_id", "")),
                "start_time": start_time,
                "end_time": end_time,
            }
        )
    return payloads


def build_retrieval_eval_template(
    segments: pd.DataFrame,
    specs: pd.DataFrame,
    max_positives: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, spec in specs.iterrows():
        recipe_keywords = _split_keywords(spec.get("recipe_keywords", ""))
        scene_keywords = _split_keywords(spec.get("scene_keywords", ""))
        target_video_id = str(spec.get("target_video_id", "") or "").strip()

        recipe_mask = _keyword_mask(segments, recipe_keywords, RECIPE_TEXT_FIELDS)
        scene_mask = _keyword_mask(segments, scene_keywords, SCENE_TEXT_FIELDS)
        matched = segments[recipe_mask & scene_mask].copy()
        if target_video_id:
            matched = matched[matched["video_id"].astype(str) == target_video_id]

        matched = matched.sort_values(["video_id", "start_time"], kind="stable")
        positive_segments = _segment_payloads(matched, max_positives)
        matched_video_ids = sorted(
            {
                str(video_id)
                for video_id in matched.get("video_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
                if video_id
            }
        )
        positive_video_ids = matched_video_ids if str(spec.get("expected_result_type", "")) == "video" else sorted(
            {item["video_id"] for item in positive_segments if item["video_id"]}
        )

        needs_review = True
        if not positive_segments and str(spec.get("expected_result_type", "")) != "video":
            note_suffix = "No automatic segment match; fill positives manually."
        elif str(spec.get("query_type", "")) == "visual_state":
            note_suffix = "Visual-state query; verify frames manually."
        else:
            note_suffix = "Auto-generated draft; verify before reporting metrics."

        notes = str(spec.get("notes", "") or "").strip()
        if note_suffix:
            notes = f"{notes} {note_suffix}".strip()

        rows.append(
            {
                "query": str(spec.get("query", "")),
                "query_type": str(spec.get("query_type", "")),
                "expected_intent": str(spec.get("expected_intent", "")),
                "expected_result_type": str(spec.get("expected_result_type", "")),
                "positive_segments": json.dumps(positive_segments, ensure_ascii=False),
                "positive_video_ids": json.dumps(positive_video_ids, ensure_ascii=False),
                "target_video_id": target_video_id,
                "auto_match_count": int(len(matched)),
                "needs_review": needs_review,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a draft retrieval evaluation CSV from canonical segments.")
    parser.add_argument("--segments", required=True, help="Path to canonical_segments.parquet or master_keyframe_dataset2.json")
    parser.add_argument("--urls", default="", help="Optional shorts_urls.csv for recipe names when using raw JSON")
    parser.add_argument("--spec", default="", help="Optional query spec CSV. If omitted, specs are built from data.")
    parser.add_argument("--output", default="data/eval/retrieval_eval_queries_draft.csv")
    parser.add_argument("--max-positives", type=int, default=20)
    args = parser.parse_args()

    segments = load_segments(args.segments, args.urls or None)
    specs = load_query_specs(args.spec) if args.spec else build_query_specs_from_segments(segments)
    result = build_retrieval_eval_template(segments, specs, args.max_positives)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(result)} draft queries to {output_path}")


if __name__ == "__main__":
    main()
