from __future__ import annotations

import os

from src.search.unified_search import UnifiedSearchResult


NO_SCENES_MESSAGE = "\uac80\uc0c9\ub41c \uc7a5\uba74\uc774 \uc5c6\uc5b4 \ub2f5\ubcc0\uc744 \uc0dd\uc131\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4."
FALLBACK_HEADER = "\uac80\uc0c9\ub41c \uc7a5\uba74 \uae30\uc900\uc73c\ub85c \ud655\uc778\ub41c \ub0b4\uc6a9\uc785\ub2c8\ub2e4."


def _fallback_answer(result: UnifiedSearchResult, reason: str = "") -> str:
    if not result.scenes:
        return NO_SCENES_MESSAGE
    lines = [FALLBACK_HEADER]
    if reason:
        lines.append(f"\ucc38\uace0: {reason}")
    for idx, scene in enumerate(result.scenes[:5], start=1):
        payload = scene.payload
        lines.append(
            (
                f"{idx}. {payload.get('recipe_name', '')} / {payload.get('video_id', '')} "
                f"({payload.get('start_time', '')}-{payload.get('end_time', '')}s): "
                f"{payload.get('caption', '')}"
            )
        )
    return "\n".join(lines)


def generate_answer(
    query: str,
    result: UnifiedSearchResult,
    model: str = "gemini-2.5-flash-lite",
) -> str:
    if result.result_type != "summary":
        return ""
    if not result.scenes:
        return _fallback_answer(result)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_answer(result, "GEMINI_API_KEY\uac00 \uc5c6\uc5b4 \uac80\uc0c9 context\ub9cc \ud45c\uc2dc\ud569\ub2c8\ub2e4.")

    context = result.answer_context or _fallback_answer(result)
    prompt = f"""
You are an answer generator for a Korean cooking shorts search system.
Answer in Korean only, using only the retrieved scene context below.
Do not guess ingredients, quantities, or steps that are not present in the context.
If something is not supported by the context, say "\ud655\uc778\ub418\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4".

User query:
{query}

Retrieved context:
{context}

Output format:
- 1-2 sentence summary
- Confirmed ingredients/actions/steps
- Evidence timestamps
""".strip()

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = (getattr(response, "text", "") or "").strip()
        return text or _fallback_answer(result, "Gemini \uc751\ub2f5\uc774 \ube44\uc5b4 \uc788\uc5b4 \uac80\uc0c9 context\ub9cc \ud45c\uc2dc\ud569\ub2c8\ub2e4.")
    except Exception as exc:
        return _fallback_answer(result, f"Gemini \ub2f5\ubcc0 \uc0dd\uc131 \uc2e4\ud328: {type(exc).__name__}: {exc}")
