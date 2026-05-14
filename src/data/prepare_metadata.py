from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "segment_id",
    "video_id",
    "recipe_name",
    "caption",
    "start_time",
    "end_time",
    "current_time",
    "frame_path",
    "video_path",
    "youtube_url",
]


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON file expected to contain a list of segment records."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}")
    return [dict(item) for item in data]


def load_url_metadata(path: str | Path) -> pd.DataFrame:
    """Load shorts URL metadata and normalize column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename_map = {
        "memo": "recipe_name",
        "url": "youtube_url",
    }
    df = df.rename(columns=rename_map)
    required = {"video_id", "youtube_url", "recipe_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"URL metadata missing columns: {sorted(missing)}")
    return df[["video_id", "youtube_url", "recipe_name"]]


def _resolve_dataset_path(dataset_root: Path, raw_path: str | None) -> str:
    if not raw_path:
        return ""
    raw_path = raw_path.replace("\\", "/")
    marker = "korean_cooking_shorts_dataset/"
    if marker in raw_path:
        rel = raw_path.split(marker, 1)[1]
        return str(dataset_root / rel)
    if raw_path.startswith(("frames/", "videos/", "metadata/", "urls/")):
        return str(dataset_root / raw_path)
    return raw_path


def _frame_sort_key(path: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "_", path).strip("_")


def build_segments(
    master_json: str | Path,
    urls_csv: str | Path,
    dataset_root: str | Path,
) -> pd.DataFrame:
    """Build canonical segment metadata from keyframe JSON and URL CSV."""
    dataset_root = Path(dataset_root)
    records = load_json_records(master_json)
    urls = load_url_metadata(urls_csv)

    rows: list[dict[str, Any]] = []
    for rec in records:
        video_id = str(rec.get("video_id", "")).strip()
        if not video_id:
            continue
        frame_path = _resolve_dataset_path(dataset_root, rec.get("image_path"))
        video_path = str(dataset_root / "videos" / f"{video_id}.mp4")
        current_time = float(rec.get("current_time", rec.get("timestamp_sec", 0.0)) or 0.0)
        start_time = float(rec.get("start_time", current_time) or current_time)
        end_time = float(rec.get("end_time", current_time) or current_time)
        caption = str(rec.get("caption", "")).strip()
        segment_id = f"{video_id}_{int(round(current_time * 1000)):08d}_{_frame_sort_key(Path(frame_path).stem)}"
        rows.append(
            {
                "segment_id": segment_id,
                "video_id": video_id,
                "caption": caption,
                "start_time": start_time,
                "end_time": end_time,
                "current_time": current_time,
                "frame_path": frame_path,
                "video_path": video_path,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No segment rows were produced from metadata")

    df = df.merge(urls, on="video_id", how="left")
    df["recipe_name"] = df["recipe_name"].fillna("")
    df["youtube_url"] = df["youtube_url"].fillna("")
    df = df[REQUIRED_COLUMNS]
    return df.sort_values(["video_id", "current_time", "segment_id"]).reset_index(drop=True)


def validate_segments(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    null_cols = [col for col in REQUIRED_COLUMNS if df[col].isna().any()]
    if null_cols:
        raise ValueError(f"Null values found in required columns: {null_cols}")
    if df["segment_id"].duplicated().any():
        dupes = df.loc[df["segment_id"].duplicated(), "segment_id"].head().tolist()
        raise ValueError(f"Duplicate segment_id values found, examples: {dupes}")


def write_segments(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validate_segments(df)
    df.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical MVP segment metadata")
    parser.add_argument("--master-json", required=True)
    parser.add_argument("--urls-csv", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = build_segments(args.master_json, args.urls_csv, args.dataset_root)
    write_segments(df, args.output)
    print(f"Wrote {len(df)} segments to {args.output}")


if __name__ == "__main__":
    main()
