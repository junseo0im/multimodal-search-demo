from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qdrant_client import QdrantClient

from src.index.build_index import COLLECTION_NAME
from src.models.bge_encoder import BGEEncoder
from src.models.siglip_encoder import SigLIPEncoder
from src.search.dedup import dedup_adjacent
from src.search.fusion import ScoredCandidate
from src.search.hybrid_search import hybrid_search, text_search
from src.search.query_analyzer import QueryPlan, analyze_query


@dataclass
class VideoCandidate:
    video_id: str
    recipe_name: str
    score: float
    scene_count: int = 1
    youtube_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedSearchResult:
    plan: QueryPlan
    scenes: list[ScoredCandidate]
    videos: list[VideoCandidate]
    answer_context: str = ""


def _video_candidates_from_results(results: list[Any], limit: int = 5) -> list[VideoCandidate]:
    grouped: dict[str, VideoCandidate] = {}
    for item in results:
        payload = dict(item.payload or {})
        video_id = str(payload.get("video_id", ""))
        if not video_id:
            continue
        score = float(getattr(item, "score", 0.0))
        current = grouped.get(video_id)
        if current is None:
            grouped[video_id] = VideoCandidate(
                video_id=video_id,
                recipe_name=str(payload.get("recipe_name", "")),
                score=score,
                scene_count=1,
                youtube_url=str(payload.get("youtube_url", "")),
            )
        else:
            current.score = max(current.score, score)
            current.scene_count += 1
            if not current.youtube_url:
                current.youtube_url = str(payload.get("youtube_url", ""))
    return sorted(grouped.values(), key=lambda item: item.score, reverse=True)[:limit]


def _video_candidates_from_scenes(scenes: list[ScoredCandidate], limit: int = 5) -> list[VideoCandidate]:
    grouped: dict[str, VideoCandidate] = {}
    for item in scenes:
        payload = item.payload
        video_id = str(payload.get("video_id", ""))
        if not video_id:
            continue
        current = grouped.get(video_id)
        if current is None:
            grouped[video_id] = VideoCandidate(
                video_id=video_id,
                recipe_name=str(payload.get("recipe_name", "")),
                score=float(item.hybrid_score),
                scene_count=1,
                youtube_url=str(payload.get("youtube_url", "")),
            )
        else:
            current.score = max(current.score, float(item.hybrid_score))
            current.scene_count += 1
    return sorted(grouped.values(), key=lambda item: item.score, reverse=True)[:limit]


def _answer_context(plan: QueryPlan, scenes: list[ScoredCandidate]) -> str:
    if not plan.needs_generation:
        return ""
    lines = ["Generation-ready context from retrieved scenes:"]
    for idx, scene in enumerate(scenes[:8], start=1):
        p = scene.payload
        lines.append(
            (
                f"{idx}. video_id={p.get('video_id', '')}, recipe={p.get('recipe_name', '')}, "
                f"time={p.get('start_time', '')}-{p.get('end_time', '')}, caption={p.get('caption', '')}"
            )
        )
    return "\n".join(lines)


def _merge_scene_results(groups: list[list[ScoredCandidate]], top_k: int) -> list[ScoredCandidate]:
    merged: list[ScoredCandidate] = []
    for group in groups:
        merged.extend(group)
    merged.sort(key=lambda item: item.hybrid_score, reverse=True)
    return dedup_adjacent(merged, top_k=top_k)


def unified_search(
    client: QdrantClient,
    bge: BGEEncoder,
    siglip: SigLIPEncoder,
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    optional_video_id: str | None = None,
) -> UnifiedSearchResult:
    plan = analyze_query(query, optional_video_id)
    weights = plan.weights
    scene_query = plan.scene_query or query
    video_filter = optional_video_id.strip() if optional_video_id else None

    if video_filter:
        scenes = hybrid_search(
            client,
            bge,
            siglip,
            scene_query,
            collection_name,
            top_k=top_k,
            alpha=weights.text,
            beta=weights.image,
            video_id=video_filter,
        )
        videos = _video_candidates_from_scenes(scenes)
        return UnifiedSearchResult(plan=plan, scenes=scenes, videos=videos, answer_context=_answer_context(plan, scenes))

    if plan.intent == "video_search":
        raw = text_search(client, bge, plan.video_query or query, collection_name, top_n=max(top_k * 4, 20))
        videos = _video_candidates_from_results(raw, limit=top_k)
        scenes = hybrid_search(
            client,
            bge,
            siglip,
            plan.video_query or query,
            collection_name,
            top_k=top_k,
            alpha=weights.text,
            beta=weights.image,
        )
        return UnifiedSearchResult(plan=plan, scenes=scenes, videos=videos)

    if plan.intent == "compound_scene_search" and plan.scope == "video_candidate":
        raw_videos = text_search(client, bge, plan.video_query or query, collection_name, top_n=30)
        videos = _video_candidates_from_results(raw_videos, limit=3)
        scene_groups = [
            hybrid_search(
                client,
                bge,
                siglip,
                scene_query,
                collection_name,
                top_k=top_k,
                alpha=weights.text,
                beta=weights.image,
                video_id=video.video_id,
            )
            for video in videos
        ]
        scenes = _merge_scene_results(scene_groups, top_k)
        return UnifiedSearchResult(plan=plan, scenes=scenes, videos=videos, answer_context=_answer_context(plan, scenes))

    scenes = hybrid_search(
        client,
        bge,
        siglip,
        scene_query,
        collection_name,
        top_k=top_k,
        alpha=weights.text,
        beta=weights.image,
    )
    videos = _video_candidates_from_scenes(scenes)
    return UnifiedSearchResult(plan=plan, scenes=scenes, videos=videos, answer_context=_answer_context(plan, scenes))
