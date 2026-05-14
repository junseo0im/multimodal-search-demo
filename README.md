# Cooking Shorts Multimodal Search

A scene-level search demo for Korean cooking short-form videos. The system indexes short videos as searchable scene segments, then combines text retrieval and visual retrieval to find the most relevant moment for a natural-language query.

## Overview

Short-form cooking videos often contain useful information in very small time windows: ingredient preparation, seasoning, cooking actions, and final plating. A saved video list can tell you which videos you have, but it usually cannot answer questions such as "where does the cook add green onion?" or "which moment shows the finished dish?"

This project supports two related retrieval tasks:

- **Across-video search**: find relevant videos or scenes across the whole collection.
- **In-video search**: restrict the search to a specific `video_id` and find the relevant moment inside that video.

The first target domain is Korean cooking videos because cooking has clear temporal structure and many practical scene-level queries.

## Pipeline

```text
metadata JSON / URL CSV
  -> canonical scene segment metadata
  -> fine-tuned SigLIP2 image embeddings
  -> BGE-M3 text embeddings
  -> Qdrant Cloud index
  -> Gradio search demo
```

Each Qdrant point represents one scene segment or representative keyframe. The payload stores the video id, caption, time range, frame path, video path, recipe name, and YouTube URL.

## Retrieval

### Image-side retrieval

- Base model: `google/siglip2-base-patch16-224`
- Adapter: cooking-domain LoRA adapter
- Purpose: embed representative frames and compare them with the query encoded by the SigLIP2 text encoder.

### Text-side retrieval

- Model: `BAAI/bge-m3`
- Input text: `{recipe_name}. {caption}`
- Purpose: retrieve scenes using semantic text similarity over recipe names and scene captions.

### Hybrid retrieval

The demo exposes three modes:

- `text-only`
- `image-only`
- `hybrid`

Hybrid search normalizes text and image scores separately, then combines them with a weighted sum. The current implementation also keeps the individual scores visible so search behavior can be inspected during demos.

## Repository Structure

```text
notebooks/
  01_prepare_metadata.ipynb
  02_build_qdrant_index.ipynb
  03_gradio_demo.ipynb
  04_run_eval.ipynb

src/
  data/      metadata preparation
  models/    BGE-M3 and SigLIP2 encoder wrappers
  index/     Qdrant Cloud indexing
  search/    text/image/hybrid search, fusion, deduplication
  ui/        Gradio demo
  eval/      evaluation metrics and runner

tests/
  lightweight unit tests for ranking and metric helpers
```

## Setup

The notebooks are designed for Colab.

```bash
pip install -r requirements-colab.txt
```

Set these Colab Secrets:

```text
QDRANT_URL
QDRANT_API_KEY
```

Expected Google Drive layout:

```text
MyDrive/
  korean_cooking_shorts_dataset/
    videos/
    frames/
    metadata/
      master_keyframe_dataset2.json
    urls/
      shorts_urls.csv
    siglip2_lora_qv_r16_best/
```

If your Drive layout differs, update the path variables in the notebooks.

## Running

For the first run:

```text
01_prepare_metadata.ipynb
02_build_qdrant_index.ipynb
03_gradio_demo.ipynb
```

After the Qdrant index has been built, normal demo iteration only needs:

```text
03_gradio_demo.ipynb
```

The Gradio demo includes:

- search mode selection
- whole-collection search or `video_id`-filtered search
- Top-K result table
- representative frame gallery
- Top-1 preview clip generated from the Drive mp4

## Evaluation

`04_run_eval.ipynb` is prepared for pipeline-level evaluation using query records with positive time ranges.

Planned metrics:

- Recall@K
- MRR
- temporal mIoU

The main comparison is:

```text
text-only vs image-only vs hybrid
```

## Notes

- Qdrant Cloud is used as a rebuildable search index, not as the metadata source of truth.
- Source metadata remains in Google Drive JSON/CSV/Parquet files.
- Videos, frames, adapters, generated embeddings, and other large artifacts should not be committed.
- The indexing notebook can recreate the Qdrant collection; the demo notebook only reads from the existing collection.

