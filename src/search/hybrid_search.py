from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from src.index.build_index import COLLECTION_NAME, IMAGE_VECTOR, TEXT_VECTOR
from src.models.bge_encoder import BGEEncoder
from src.models.siglip_encoder import SigLIPEncoder
from src.search.dedup import dedup_adjacent
from src.search.fusion import ScoredCandidate, fuse_results, weights_for_query


def _video_filter(video_id: str | None) -> models.Filter | None:
    if not video_id:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
    )


def text_search(
    client: QdrantClient,
    bge: BGEEncoder,
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_n: int = 20,
    video_id: str | None = None,
) -> list[Any]:
    return client.query_points(
        collection_name=collection_name,
        query=bge.encode(query).tolist(),
        using=TEXT_VECTOR,
        query_filter=_video_filter(video_id),
        limit=top_n,
        with_payload=True,
    ).points


def image_search(
    client: QdrantClient,
    siglip: SigLIPEncoder,
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_n: int = 20,
    video_id: str | None = None,
) -> list[Any]:
    return client.query_points(
        collection_name=collection_name,
        query=siglip.encode_texts(query).tolist(),
        using=IMAGE_VECTOR,
        query_filter=_video_filter(video_id),
        limit=top_n,
        with_payload=True,
    ).points


def hybrid_search(
    client: QdrantClient,
    bge: BGEEncoder,
    siglip: SigLIPEncoder,
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    top_n: int = 50,
    alpha: float | None = None,
    beta: float | None = None,
    video_id: str | None = None,
) -> list[ScoredCandidate]:
    if alpha is None or beta is None:
        alpha, beta = weights_for_query(query)
    text_results = text_search(client, bge, query, collection_name, top_n, video_id)
    image_results = image_search(client, siglip, query, collection_name, top_n, video_id)
    fused = fuse_results(text_results, image_results, alpha=alpha, beta=beta)
    return dedup_adjacent(fused, top_k=top_k)

