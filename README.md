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
  -> query analyzer
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

The demo exposes a unified search flow:

- unified natural-language search
- optional `video_id` filtering
- analyzer debug for inspecting the chosen search strategy

The current implementation uses a query analyzer to choose the search intent, scope, and text/image fusion weights. Gemini can be used for query analysis when an API key is available; otherwise the demo falls back to a rule-based analyzer. The individual scores remain visible so search behavior can be inspected during demos.

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

templates/
  editable evaluation query templates
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
GEMINI_API_KEY  # optional; rule-based analyzer is used when omitted
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

- unified natural-language search
- optional `video_id`-filtered search
- Top-K result table
- representative frame gallery
- Top-1 preview clip generated from the Drive mp4
- generated answers for summary-style queries when `GEMINI_API_KEY` is available
- full-video preview near the matched timestamp when the source mp4 is accessible

## Evaluation

`templates/evaluation_queries.csv` provides a human-editable starting point for pipeline-level evaluation. Each row contains a natural-language query, expected routing labels, and positive time ranges.

Retrieval metrics:

- Recall@K
- MRR
- temporal mIoU

The main comparison is:

```text
text-only vs image-only vs hybrid vs unified
```

Run retrieval evaluation:

```bash
python -m src.eval.run_eval --queries templates/evaluation_queries.csv --adapter-path /path/to/siglip2_lora_qv_r16_best
```

Run Query Analyzer evaluation:

```bash
python -m src.eval.analyzer_eval --queries templates/evaluation_queries.csv
```

## Notes

- Qdrant Cloud is used as a rebuildable search index, not as the metadata source of truth.
- Source metadata remains in Google Drive JSON/CSV/Parquet files.
- Videos, frames, adapters, generated embeddings, and other large artifacts should not be committed.
- The indexing notebook can recreate the Qdrant collection; the demo notebook only reads from the existing collection.
