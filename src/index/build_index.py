from __future__ import annotations

import argparse
import uuid
from typing import Iterable

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from src.index.qdrant_client import get_qdrant_client
from src.models.bge_encoder import BGEEncoder
from src.models.siglip_encoder import SigLIPEncoder


COLLECTION_NAME = "cooking_segments"
IMAGE_VECTOR = "image_siglip"
TEXT_VECTOR = "text_bge"
OPTIONAL_TEXT_FIELDS = ("title_text", "asr_text", "ocr_text", "scene_caption")


def stable_point_id(segment_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cooking-segment:{segment_id}"))


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def make_search_text(row: pd.Series) -> str:
    parts = [_clean_text(row.get("recipe_name", ""))]
    for field in OPTIONAL_TEXT_FIELDS:
        value = _clean_text(row.get(field, ""))
        if value:
            parts.append(value)
    caption = _clean_text(row.get("scene_caption", "")) or _clean_text(row.get("caption", ""))
    if caption and caption not in parts:
        parts.append(caption)
    return ". ".join(part for part in parts if part).strip(". ")


def create_collection(
    client: QdrantClient,
    collection_name: str,
    image_dim: int,
    text_dim: int,
    recreate: bool = False,
) -> None:
    if client.collection_exists(collection_name):
        if not recreate:
            return
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            IMAGE_VECTOR: models.VectorParams(size=image_dim, distance=models.Distance.COSINE),
            TEXT_VECTOR: models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
        },
    )
    for field in ("video_id", "recipe_name"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def build_and_upsert(
    segments: pd.DataFrame,
    client: QdrantClient,
    siglip: SigLIPEncoder,
    bge: BGEEncoder,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
    batch_size: int = 32,
) -> None:
    first = segments.iloc[:1]
    image_dim = int(siglip.encode_images(first["frame_path"].tolist()).shape[-1])
    text_dim = int(bge.encode([make_search_text(first.iloc[0])]).shape[-1])
    create_collection(client, collection_name, image_dim=image_dim, text_dim=text_dim, recreate=recreate)

    for batch_df in iter_batches(segments, batch_size):
        image_vecs = siglip.encode_images(batch_df["frame_path"].tolist(), batch_size=batch_size)
        text_vecs = bge.encode([make_search_text(row) for _, row in batch_df.iterrows()])
        points = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            payload = {
                "segment_id": row["segment_id"],
                "video_id": row["video_id"],
                "recipe_name": row["recipe_name"],
                "caption": row["caption"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "current_time": float(row["current_time"]),
                "frame_path": row["frame_path"],
                "video_path": row["video_path"],
                "youtube_url": row["youtube_url"],
            }
            for field in OPTIONAL_TEXT_FIELDS:
                value = _clean_text(row.get(field, ""))
                if value:
                    payload[field] = value
            points.append(
                models.PointStruct(
                    id=stable_point_id(str(row["segment_id"])),
                    vector={
                        IMAGE_VECTOR: np.asarray(image_vecs[idx], dtype=np.float32).tolist(),
                        TEXT_VECTOR: np.asarray(text_vecs[idx], dtype=np.float32).tolist(),
                    },
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted {len(points)} points")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Qdrant MVP index")
    parser.add_argument("--segments", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_parquet(args.segments)
    client = get_qdrant_client()
    siglip = SigLIPEncoder(adapter_path=args.adapter_path)
    bge = BGEEncoder()
    build_and_upsert(df, client, siglip, bge, args.collection, args.recreate, args.batch_size)


if __name__ == "__main__":
    main()
