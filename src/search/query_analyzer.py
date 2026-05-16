from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal


Intent = Literal["video_search", "scene_search", "in_video_search", "compound_scene_search", "summary"]
Scope = Literal["global", "video_id", "video_candidate"]
QueryType = Literal["recipe", "ingredient_action", "visual_state", "timing", "summary"]


@dataclass
class SearchWeights:
    text: float = 0.6
    image: float = 0.4

    def normalized(self) -> "SearchWeights":
        text = max(0.0, min(1.0, float(self.text)))
        image = max(0.0, min(1.0, float(self.image)))
        total = text + image
        if total == 0:
            return SearchWeights()
        return SearchWeights(text=text / total, image=image / total)


@dataclass
class QueryPlan:
    intent: Intent = "scene_search"
    scope: Scope = "global"
    video_query: str = ""
    scene_query: str = ""
    query_type: QueryType = "ingredient_action"
    weights: SearchWeights | None = None
    needs_generation: bool = False
    generation_task: str | None = None
    analyzer: str = "rule"
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = SearchWeights()
        self.weights = self.weights.normalized()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["weights"] = asdict(self.weights)
        return data


VALID_INTENTS = {"video_search", "scene_search", "in_video_search", "compound_scene_search", "summary"}
VALID_SCOPES = {"global", "video_id", "video_candidate"}
VALID_QUERY_TYPES = {"recipe", "ingredient_action", "visual_state", "timing", "summary"}

VIDEO_TERMS = (
    "\uc601\uc0c1",
    "\uc1fc\uce20",
    "\ub808\uc2dc\ud53c",
    "\ub9cc\ub4dc\ub294",
    "\ucc3e\uc544",
)
SCENE_TERMS = (
    "\uc7a5\uba74",
    "\ubd80\ubd84",
    "\ub123",
    "\ubcf6",
    "\ub053",
    "\uc369",
    "\uc790\ub974",
    "\uc644\uc131",
)
VISUAL_TERMS = (
    "\ubcf4\uc774\ub294",
    "\uc0c9",
    "\ubaa8\uc591",
    "\uc644\uc131",
    "\ube44\uc8fc\uc5bc",
    "\ud504\ub77c\uc774\ud32c",
    "\uadf8\ub987",
    "\ub178\ub987",
    "\ube68\uac04",
)
TIMING_TERMS = (
    "\uba87 \ubd84",
    "\uc5b8\uc81c",
    "\uc21c\uc11c",
    "\uc2a4\ud47c",
    "\uc815\ub7c9",
    "\uacc4\ub7c9",
)
SUMMARY_TERMS = (
    "\uc694\uc57d",
    "\uc815\ub9ac",
    "\uc124\uba85",
    "\ucd94\ucc9c",
    "\uc7ac\ub8cc",
    "\uc21c\uc11c",
    "\uc54c\ub824\uc918",
)
IN_VIDEO_TERMS = ("\uc774 \uc601\uc0c1", "\uc548\uc5d0\uc11c", "\uc601\uc0c1\uc5d0\uc11c", "video_id", "short_")
AMBIGUOUS_CONTEXT_TERMS = ("\uadf8 \uc7a5\uba74", "\uc544\uae4c", "\uadf8\uac70", "\uc774\uac70")


def weights_for_query(query: str, query_type: QueryType | None = None) -> SearchWeights:
    if query_type == "recipe":
        return SearchWeights(0.75, 0.25)
    if query_type == "visual_state":
        return SearchWeights(0.4, 0.6)
    if query_type == "timing":
        return SearchWeights(0.85, 0.15)
    if query_type == "summary":
        return SearchWeights(0.8, 0.2)

    visual_hits = sum(term in query for term in VISUAL_TERMS)
    timing_hits = sum(term in query for term in TIMING_TERMS)
    action_hits = sum(term in query for term in SCENE_TERMS)
    if timing_hits:
        return SearchWeights(0.85, 0.15)
    if visual_hits > action_hits:
        return SearchWeights(0.4, 0.6)
    if action_hits:
        return SearchWeights(0.65, 0.35)
    return SearchWeights(0.6, 0.4)


def _contains_any(query: str, terms: tuple[str, ...]) -> bool:
    return any(term in query for term in terms)


def _split_compound_query(query: str) -> tuple[str, str]:
    for marker in ("\uc5d0\uc11c", "\uc548\uc5d0\uc11c"):
        if marker in query:
            left, right = query.split(marker, 1)
            return left.strip(), right.strip() or query
    return "", query


def rule_based_analyze(query: str, optional_video_id: str | None = None, reason: str = "") -> QueryPlan:
    normalized = query.strip()
    has_video_id = bool(optional_video_id) or bool(re.search(r"short_\d+", normalized))
    wants_summary = _contains_any(normalized, SUMMARY_TERMS) and not _contains_any(normalized, SCENE_TERMS)
    has_video_terms = _contains_any(normalized, VIDEO_TERMS)
    has_scene_terms = _contains_any(normalized, SCENE_TERMS)
    has_in_video_terms = has_video_id or _contains_any(normalized, IN_VIDEO_TERMS)
    has_ambiguous_context = _contains_any(normalized, AMBIGUOUS_CONTEXT_TERMS)
    has_visual_terms = _contains_any(normalized, VISUAL_TERMS)
    has_timing_terms = _contains_any(normalized, TIMING_TERMS)

    query_type: QueryType = "ingredient_action"
    if wants_summary:
        query_type = "summary"
    elif has_timing_terms:
        query_type = "timing"
    elif has_visual_terms and not has_scene_terms:
        query_type = "visual_state"
    elif has_video_terms and not has_scene_terms:
        query_type = "recipe"

    if wants_summary:
        intent: Intent = "summary"
        scope: Scope = "video_id" if has_in_video_terms else "global"
        video_query, scene_query = _split_compound_query(normalized)
    elif has_video_terms and has_scene_terms:
        intent = "compound_scene_search"
        scope = "video_id" if has_video_id else "video_candidate"
        video_query, scene_query = _split_compound_query(normalized)
    elif (has_in_video_terms or has_ambiguous_context) and has_scene_terms:
        intent = "in_video_search"
        scope = "video_id" if has_video_id else "video_candidate"
        video_query, scene_query = _split_compound_query(normalized)
    elif has_video_terms and not has_scene_terms:
        intent = "video_search"
        scope = "global"
        video_query, scene_query = normalized, normalized
    else:
        intent = "scene_search"
        scope = "video_id" if has_video_id else "global"
        video_query, scene_query = "", normalized

    if not scene_query:
        scene_query = normalized
    if not video_query and intent in {"video_search", "compound_scene_search"}:
        video_query = normalized

    return QueryPlan(
        intent=intent,
        scope=scope,
        video_query=video_query,
        scene_query=scene_query,
        query_type=query_type,
        weights=weights_for_query(normalized, query_type),
        needs_generation=intent == "summary",
        generation_task="summary_context" if intent == "summary" else None,
        analyzer="rule",
        fallback_reason=reason,
    )


def _coerce_plan(data: dict[str, Any], query: str, optional_video_id: str | None) -> QueryPlan:
    fallback = rule_based_analyze(query, optional_video_id)
    weights = data.get("weights") if isinstance(data.get("weights"), dict) else {}
    intent = data.get("intent", fallback.intent)
    scope = data.get("scope", fallback.scope)
    query_type = data.get("query_type", fallback.query_type)

    plan = QueryPlan(
        intent=intent if intent in VALID_INTENTS else fallback.intent,
        scope=scope if scope in VALID_SCOPES else fallback.scope,
        video_query=str(data.get("video_query") or fallback.video_query or ""),
        scene_query=str(data.get("scene_query") or fallback.scene_query or query),
        query_type=query_type if query_type in VALID_QUERY_TYPES else fallback.query_type,
        weights=SearchWeights(
            text=float(weights.get("text", fallback.weights.text)),
            image=float(weights.get("image", fallback.weights.image)),
        ),
        needs_generation=bool(data.get("needs_generation", fallback.needs_generation)),
        generation_task=data.get("generation_task") or fallback.generation_task,
        analyzer="gemini",
    )
    return plan


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def gemini_analyze(
    query: str,
    optional_video_id: str | None = None,
    model: str = "gemini-2.5-flash-lite",
) -> QueryPlan:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return rule_based_analyze(query, optional_video_id, "GEMINI_API_KEY is not set.")

    prompt = f"""
You are a query planner for a Korean cooking shorts multimodal search system.
Return only valid JSON. Do not add markdown.

Allowed intent values: video_search, scene_search, in_video_search, compound_scene_search, summary.
Allowed scope values: global, video_id, video_candidate.
Allowed query_type values: recipe, ingredient_action, visual_state, timing, summary.

Rules:
- Use compound_scene_search when the query asks for a scene inside a named recipe/video.
- Use scene_search for ingredient/action/visual moment searches across all videos.
- Use video_search when the user mainly wants a relevant video.
- Use summary when the user asks to summarize, explain, recommend, or list ingredients.
- If the user says "this video", "that scene", or "earlier" without a video_id, use video_candidate scope rather than inventing an id.
- Cooking ingredient/action queries often rely on text, ASR, captions, or OCR.
- Visual state queries should increase image weight.

User query: {query}
Optional video_id: {optional_video_id or ""}

JSON schema:
{{
  "intent": "...",
  "scope": "...",
  "video_query": "...",
  "scene_query": "...",
  "query_type": "...",
  "weights": {{"text": 0.6, "image": 0.4}},
  "needs_generation": false,
  "generation_task": null
}}
""".strip()

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = getattr(response, "text", "") or ""
        return _coerce_plan(_extract_json(text), query, optional_video_id)
    except Exception as exc:
        return rule_based_analyze(query, optional_video_id, f"Gemini analyzer failed: {type(exc).__name__}: {exc}")


def analyze_query(query: str, optional_video_id: str | None = None) -> QueryPlan:
    return gemini_analyze(query, optional_video_id)
