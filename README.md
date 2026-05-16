# Cooking Shorts Multimodal Search

한국어 요리 쇼츠 영상을 대상으로 한 멀티모달 장면 검색 데모입니다. 수집한 쇼츠를 영상 단위가 아니라 검색 가능한 scene segment 단위로 인덱싱하고, 텍스트 검색과 이미지 검색을 함께 사용해 사용자의 자연어 질의에 맞는 영상 또는 장면을 찾아줍니다.

## 프로젝트 개요

요리 쇼츠는 재료 손질, 양념 투입, 조리 동작, 완성 장면처럼 짧은 시간 구간에 중요한 정보가 모여 있습니다. 단순히 저장된 영상 목록만으로는 “대파 넣는 장면”, “완성된 장면”, “돌솥비빔밥 영상에서 계란 넣는 부분” 같은 질의에 답하기 어렵습니다.

이 프로젝트는 두 가지 검색 상황을 함께 다룹니다.

- **영상 간 검색**: 전체 쇼츠 컬렉션에서 관련 영상 또는 장면을 찾습니다.
- **영상 내 검색**: 특정 `video_id` 안에서 사용자가 원하는 장면을 찾습니다.

현재 1차 도메인은 한국어 요리 쇼츠입니다. 요리 영상은 절차와 장면 구분이 비교적 명확하고, 실제 사용자가 장면 단위로 찾고 싶은 요구가 많기 때문에 검색 데모와 평가에 적합합니다.

## 전체 파이프라인

```text
metadata JSON / URL CSV
  -> canonical scene segment metadata
  -> fine-tuned SigLIP2 image embeddings
  -> BGE-M3 text embeddings
  -> Qdrant Cloud index
  -> query analyzer
  -> Gradio search demo
```

Qdrant에는 `1 point = 1 scene segment/keyframe` 구조로 저장합니다. payload에는 `video_id`, `recipe_name`, `caption`, 시간 구간, 대표 프레임 경로, 원본 영상 경로, YouTube URL 등 데모와 평가에 필요한 최소 정보를 둡니다.

## 검색 구조

### 이미지 기반 검색

- Base model: `google/siglip2-base-patch16-224`
- Adapter: Recipe1M 기반으로 학습한 cooking-domain LoRA adapter
- 역할: 대표 프레임 이미지를 임베딩하고, 질의는 SigLIP2 text encoder로 임베딩해 이미지 측면의 관련 장면을 찾습니다.

### 텍스트 기반 검색

- Model: `BAAI/bge-m3`
- 입력 텍스트: `recipe_name`, `caption`, 선택적으로 `title_text`, `asr_text`, `ocr_text`, `scene_caption`
- 역할: 영상 제목/음식명/자막/캡션에 드러나는 조리 정보와 의미적으로 가까운 장면을 찾습니다.

### 하이브리드 검색

현재 데모는 사용자가 검색 모드를 직접 고르지 않아도 되는 통합 검색 흐름을 사용합니다.

- 자연어 질의 입력
- 선택적 `video_id` 필터
- Query Analyzer를 통한 intent/scope/weight 결정
- text/image 검색 결과 fusion
- 장면 중복 제거 및 영상 후보 집계

`GEMINI_API_KEY`가 있으면 Gemini 기반 Query Analyzer와 summary 답변 생성을 사용할 수 있습니다. API key가 없으면 rule-based fallback으로 검색 데모는 계속 동작합니다.

## 저장소 구조

```text
notebooks/
  01_prepare_metadata.ipynb
  02_build_qdrant_index.ipynb
  03_gradio_demo.ipynb
  04_run_eval.ipynb

src/
  data/        metadata preparation
  models/      BGE-M3 / SigLIP2 encoder wrappers
  index/       Qdrant Cloud indexing
  search/      text/image/hybrid search, fusion, query analyzer
  generation/  Gemini answer generation
  ui/          Gradio demo
  eval/        evaluation dataset builder, metrics, runner

templates/
  evaluation query templates and dataset-derived draft queries

tests/
  lightweight unit tests
```

## 실행 환경

노트북은 Colab 기준으로 작성되어 있습니다.

```bash
pip install -r requirements-colab.txt
```

Colab Secrets에는 다음 값을 설정합니다.

```text
QDRANT_URL
QDRANT_API_KEY
GEMINI_API_KEY  # optional
```

기본 Google Drive 구조는 다음을 가정합니다.

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

경로가 다르면 각 노트북의 path 변수만 수정하면 됩니다.

## 실행 순서

처음 인덱스를 만들 때는 아래 순서로 실행합니다.

```text
01_prepare_metadata.ipynb
02_build_qdrant_index.ipynb
03_gradio_demo.ipynb
```

Qdrant 인덱스가 이미 만들어진 뒤에는 보통 데모 노트북만 실행하면 됩니다.

```text
03_gradio_demo.ipynb
```

Gradio 데모에서 제공하는 주요 기능은 다음과 같습니다.

- 통합 자연어 검색
- 특정 `video_id` 내부 검색
- Top-K 결과 테이블
- 대표 프레임 갤러리
- Top-1 장면 clip preview
- 원본 영상 timestamp preview
- summary 질의에 대한 Gemini 기반 답변 생성
- Analyzer Debug 확인

## 평가

정량 평가는 안정적인 정답을 만들 수 있는 검색 질의로 범위를 제한합니다.

- 영상 검색: 예를 들어 `돌솥비빔밥 영상 찾아줘`
- 재료/동작 장면 검색: 예를 들어 `대파 넣는 장면`
- 시각 상태 장면 검색: 예를 들어 `완성된 장면`
- 복합 검색: 예를 들어 `돌솥비빔밥 영상에서 계란 넣는 장면`

요약, 추천, 후속 대화, “그 장면”처럼 context가 필요한 질의는 정량 검색 평가에 섞지 않고 데모에서 정성적으로 확인합니다.

### 평가셋 생성

기본적으로 `canonical_segments.parquet`를 읽어서 실제 데이터셋에 등장하는 음식명과 caption 기반으로 평가 질의와 정답 후보를 생성합니다.

```bash
python -m src.eval.build_retrieval_eval_template \
  --segments /path/to/canonical_segments.parquet \
  --output data/eval/retrieval_eval_queries_draft.csv
```

raw JSON/CSV만 있을 때는 아래처럼 실행할 수 있습니다.

```bash
python -m src.eval.build_retrieval_eval_template \
  --segments /path/to/master_keyframe_dataset2.json \
  --urls /path/to/shorts_urls.csv \
  --output data/eval/retrieval_eval_queries_draft.csv
```

현재 저장소의 `templates/retrieval_query_specs.csv`와 `templates/retrieval_eval_queries_draft.csv`는 200개 쇼츠 데이터셋에서 자동 생성한 시작점입니다. 최종 보고서용 수치를 내기 전에는 `positive_segments`, `positive_video_ids`를 사람이 검수하는 것을 권장합니다.

### 평가 지표

- `Recall@K`
- `MRR`
- scene-level query: temporal mIoU
- video-level query: video-id hit

비교 대상은 다음 네 가지입니다.

```text
text-only vs image-only vs hybrid vs unified
```

검색 파이프라인 평가:

```bash
python -m src.eval.run_eval \
  --queries data/eval/retrieval_eval_queries_draft.csv \
  --adapter-path /path/to/siglip2_lora_qv_r16_best \
  --output-csv data/eval/retrieval_eval_results.csv
```

Query Analyzer 평가:

```bash
python -m src.eval.analyzer_eval \
  --queries data/eval/retrieval_eval_queries_draft.csv \
  --output-csv data/eval/analyzer_eval_results.csv
```

## 설계 메모

- Qdrant Cloud는 재생성 가능한 검색 인덱스로 사용하며, 원본 metadata의 source of truth는 Drive의 JSON/CSV/Parquet 파일입니다.
- 영상/프레임/LoRA adapter/embedding 등 큰 파일은 Git에 올리지 않습니다.
- 현재 평가와 검색은 segment 단위를 기본으로 합니다. 요리 쇼츠는 “넣는 장면”, “볶는 장면”, “완성 장면”처럼 순간 검색 요구가 많기 때문입니다.
- 영상 단위 검색은 segment 검색 결과를 `video_id` 기준으로 집계해 처리합니다.
- 향후 개선 후보는 adjacent segment merge, multi-keyframe segment embedding, ASR/OCR time alignment, caption 품질 검수입니다.
