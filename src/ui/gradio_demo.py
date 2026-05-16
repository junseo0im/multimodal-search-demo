from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.index.build_index import COLLECTION_NAME
from src.index.qdrant_client import get_qdrant_client
from src.models.bge_encoder import BGEEncoder
from src.models.siglip_encoder import SigLIPEncoder
from src.search.unified_search import UnifiedSearchResult, unified_search


DISPLAY_COLUMNS = [
    "rank",
    "recipe_name",
    "video_id",
    "time",
    "caption",
    "text_score",
    "image_score",
    "hybrid_score",
    "youtube_url",
]

EXAMPLE_QUERIES = [
    "\ub300\ud30c \ub123\ub294 \uc7a5\uba74",
    "\uace0\ucd94\uc7a5 \ub123\ub294 \uc7a5\uba74",
    "\uc644\uc131\ub41c \ub5a1\uad6d \uc7a5\uba74",
    "\ud504\ub77c\uc774\ud32c\uc5d0 \ubcf6\ub294 \uc7a5\uba74",
]

DEFAULT_QUERY = "\ub300\ud30c \ub123\ub294 \uc7a5\uba74"
SCOPE_ALL = "\uc804\uccb4 \uc601\uc0c1 \uac80\uc0c9"
SCOPE_IN_VIDEO = "\ud2b9\uc815 video_id \ub0b4\ubd80 \uac80\uc0c9"
CLIP_DIR = Path("/tmp/cooking_search_clips")
FRAME_CACHE_DIR = Path("/tmp/cooking_search_frames")


def format_result(candidate: Any, rank: int) -> dict[str, Any]:
    payload = dict(candidate.payload or {})
    start_time = float(payload.get("start_time", 0.0) or 0.0)
    end_time = float(payload.get("end_time", payload.get("current_time", start_time)) or start_time)
    return {
        "rank": rank,
        "segment_id": payload.get("segment_id", ""),
        "recipe_name": payload.get("recipe_name", ""),
        "video_id": payload.get("video_id", ""),
        "time": f"{start_time:.1f}s - {end_time:.1f}s",
        "caption": payload.get("caption", ""),
        "text_score": round(float(getattr(candidate, "text_score", 0.0)), 4),
        "image_score": round(float(getattr(candidate, "image_score", 0.0)), 4),
        "hybrid_score": round(float(getattr(candidate, "hybrid_score", getattr(candidate, "score", 0.0))), 4),
        "youtube_url": payload.get("youtube_url", ""),
        "frame_path": payload.get("frame_path", ""),
        "video_path": payload.get("video_path", ""),
        "start_time": start_time,
        "end_time": end_time,
        "current_time": float(payload.get("current_time", start_time) or start_time),
    }


def make_top_card(row: dict[str, Any] | None, clip_message: str = "") -> str:
    if not row:
        return "No results."

    lines = [
        f"### Top-1: {row.get('recipe_name') or '(no recipe name)'}",
        f"- **video_id:** `{row.get('video_id', '')}`",
        f"- **time:** {row.get('time', '')}",
        f"- **caption:** {row.get('caption', '')}",
        (
            "- **score:** "
            f"hybrid `{row.get('hybrid_score', 0)}` / "
            f"text `{row.get('text_score', 0)}` / "
            f"image `{row.get('image_score', 0)}`"
        ),
    ]
    youtube_url = row.get("youtube_url")
    if youtube_url:
        lines.append(f"- **YouTube:** {youtube_url}")
    if clip_message:
        lines.append(f"- **clip:** {clip_message}")
    return "\n".join(lines)


def make_videos_card(result: UnifiedSearchResult) -> str:
    if not result.videos:
        return "No related videos."
    lines = ["### Related Videos"]
    for idx, video in enumerate(result.videos, start=1):
        title = video.recipe_name or "(no recipe name)"
        line = f"{idx}. **{title}** - `{video.video_id}` - score `{video.score:.4f}`"
        if video.scene_count:
            line += f" - scenes `{video.scene_count}`"
        if video.youtube_url:
            line += f" - {video.youtube_url}"
        lines.append(line)
    return "\n".join(lines)


def make_debug_card(result: UnifiedSearchResult) -> str:
    plan = result.plan
    lines = [
        "### Analyzer Debug",
        f"- **analyzer:** `{plan.analyzer}`",
        f"- **intent:** `{plan.intent}`",
        f"- **scope:** `{plan.scope}`",
        f"- **query_type:** `{plan.query_type}`",
        f"- **video_query:** {plan.video_query or '(none)'}",
        f"- **scene_query:** {plan.scene_query or '(none)'}",
        f"- **weights:** text `{plan.weights.text:.2f}` / image `{plan.weights.image:.2f}`",
        f"- **needs_generation:** `{plan.needs_generation}`",
    ]
    if plan.fallback_reason:
        lines.append(f"- **fallback:** {plan.fallback_reason}")
    if result.answer_context:
        lines.extend(["", "### Answer-Ready Context", "```text", result.answer_context, "```"])
    return "\n".join(lines)


def create_clip(row: dict[str, Any] | None) -> tuple[str | None, str]:
    if not row:
        return None, ""

    video_path = str(row.get("video_path", ""))
    if not video_path or not os.path.exists(video_path):
        return None, "Original mp4 was not found; showing table and frames only."

    start = max(0.0, float(row.get("start_time", 0.0)) - 1.5)
    end = float(row.get("end_time", row.get("current_time", start + 3.0))) + 1.5
    if end <= start:
        end = start + 3.0

    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    segment_id = str(row.get("segment_id", "clip")).replace("/", "_").replace("\\", "_")
    clip_path = CLIP_DIR / f"{segment_id}_{start:.1f}_{end:.1f}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        video_path,
        "-vcodec",
        "libx264",
        "-acodec",
        "aac",
        "-movflags",
        "+faststart",
        str(clip_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return None, f"Clip generation failed: {exc}"
    if result.returncode != 0 or not clip_path.exists():
        return None, "ffmpeg failed; showing table and frames only."
    return str(clip_path), "Generated a Top-1 preview clip."


def cache_frame_for_gradio(frame_path: str, rank: int) -> str | None:
    if not frame_path or not os.path.exists(frame_path):
        return None

    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source = Path(frame_path)
    target = FRAME_CACHE_DIR / f"{rank:02d}_{source.name}"
    try:
        shutil.copy2(source, target)
    except OSError:
        return None
    return str(target)


def create_app(
    collection_name: str = COLLECTION_NAME,
    adapter_path: str | None = None,
) -> gr.Blocks:
    adapter_path = adapter_path or os.getenv("SIGLIP_ADAPTER_PATH")
    if not adapter_path:
        raise ValueError("SIGLIP adapter path is required. Set SIGLIP_ADAPTER_PATH.")

    client = get_qdrant_client()
    bge = BGEEncoder()
    siglip = SigLIPEncoder(adapter_path=adapter_path)

    def run(
        query: str,
        top_k: int,
        video_id: str,
        show_debug: bool,
    ) -> tuple[str, str, pd.DataFrame, list[str], str | None, str]:
        try:
            result = unified_search(
                client,
                bge,
                siglip,
                query,
                collection_name,
                top_k=top_k,
                optional_video_id=video_id.strip() or None,
            )
            rows = [format_result(item, i + 1) for i, item in enumerate(result.scenes)]
        except Exception as exc:
            empty = pd.DataFrame(columns=DISPLAY_COLUMNS)
            return f"Search failed: `{type(exc).__name__}: {exc}`", "", empty, [], None, ""

        frames = [
            cached
            for row in rows
            if (cached := cache_frame_for_gradio(str(row.get("frame_path", "")), int(row.get("rank", 0))))
        ]
        table = pd.DataFrame(rows)
        table = pd.DataFrame(columns=DISPLAY_COLUMNS) if table.empty else table[DISPLAY_COLUMNS]

        top_row = rows[0] if rows else None
        clip_path, clip_message = create_clip(top_row)
        top_card = make_top_card(top_row, clip_message)
        videos_card = make_videos_card(result)
        debug_card = make_debug_card(result) if show_debug else ""
        return top_card, videos_card, table, frames, clip_path, debug_card

    with gr.Blocks(title="Cooking Shorts Multimodal Search") as demo:
        gr.Markdown("# Cooking Shorts Multimodal Search")
        query = gr.Textbox(label="Query", value=DEFAULT_QUERY)
        with gr.Row():
            for example in EXAMPLE_QUERIES:
                gr.Button(example).click(lambda value=example: value, outputs=query)
        with gr.Row():
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
            video_id = gr.Textbox(label="Optional video_id filter", placeholder="short_001")
            show_debug = gr.Checkbox(value=True, label="Show analyzer debug")

        button = gr.Button("Search")
        top_card = gr.Markdown(label="Top Answer")
        clip = gr.Video(label="Top-1 clip preview")
        videos_card = gr.Markdown(label="Related Videos")
        table = gr.Dataframe(label="Related Scenes", wrap=True, interactive=False)
        gallery = gr.Gallery(label="Representative frames")
        with gr.Accordion("Advanced", open=True):
            debug_card = gr.Markdown(label="Analyzer Debug")
        button.click(
            run,
            inputs=[query, top_k, video_id, show_debug],
            outputs=[top_card, videos_card, table, gallery, clip, debug_card],
        )
    return demo


if __name__ == "__main__":
    create_app().launch()
