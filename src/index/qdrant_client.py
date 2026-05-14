from __future__ import annotations

import os

from qdrant_client import QdrantClient


def get_qdrant_client(url: str | None = None, api_key: str | None = None) -> QdrantClient:
    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")
    if not url:
        raise ValueError("Qdrant URL is required. Set QDRANT_URL or pass url.")
    if not api_key:
        raise ValueError("Qdrant API key is required. Set QDRANT_API_KEY or pass api_key.")
    return QdrantClient(url=url, api_key=api_key)

