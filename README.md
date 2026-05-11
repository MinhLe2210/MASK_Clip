# OpenSDI

OpenSDI now runs a single Triton-based pipeline:

```text
input image -> Triton dedup embedding model -> Milvus dedup -> Triton classification -> Triton nfa_vit -> final label -> OpenAI vision analysis
```

All old local TensorRT service logic has been removed from the main flow. The repo now exposes one LitServe API endpoint: `/analyze`.

## Layout

```text
.
|-- Dockerfile
|-- docker-compose.yml
|-- pipeline_client.py
|-- pipeline_server.py
|-- requirements.txt
|-- model_repository/
|   |-- AI_images_detector/
|   |   |-- 1/
|   |   `-- config.pbtxt
|   |-- dedup_embedder/
|   |   |-- 1/
|   |   `-- config.pbtxt
|   `-- nfa_vit/
|       |-- 1/
|       `-- config.pbtxt
`-- src/
    |-- config.py
    |-- dedup_service.py
    |-- image_io.py
    |-- milvus_store.py
    |-- nfa_vit.py
    `-- triton_clients.py
```

## Required Services

- Triton Inference Server with three models:
  - `dedup_embedder`
  - `AI_images_detector`
  - `nfa_vit`
- Milvus
- OpenAI API access

This repo does not start Milvus for you. Run it separately or point to an existing instance.

## Environment

Minimum required variables:

```bash
export TRITON_URL=http://127.0.0.1:30000
export MILVUS_HOST=127.0.0.1
export OPENAI_API_KEY=...
```

Common model settings:

```bash
export DEDUP_MODEL_NAME=dedup_embedder
export DEDUP_INPUT_NAME=pixel_values
export DEDUP_OUTPUT_NAME=features
export DEDUP_IMAGE_SIZE=384
export EMBED_DIM=384

export CLASSIFIER_MODEL_NAME=AI_images_detector
export CLASSIFIER_INPUT_NAME=pixel_values
export CLASSIFIER_OUTPUT_NAME=logits
export CLASSIFIER_LABELS=fake,real
export CLASSIFIER_INPUT_SIZE=384

export NFA_MODEL_NAME=nfa_vit
export NFA_IMAGE_INPUT_NAME=image
export NFA_MASK_INPUT_NAME=mask
export NFA_LABEL_INPUT_NAME=label
export NFA_MASK_OUTPUT_NAME=pred_mask
export NFA_LABEL_OUTPUT_NAME=pred_label
export NFA_IMAGE_SIZE=512
export NFA_THRESHOLD=0.5
export NFA_WHITE_RATIO_THRESHOLD=0.06
export NFA_LABEL_VALUE=0.0
export NFA_PREPROCESS_MODE=resizing
```

Milvus and dedup:

```bash
export COLLECTION_NAME=ai_detector_images_deduplicate
export MILVUS_PORT=19530
export MILVUS_DATABASE=default
export DUP_THRESHOLD=0.999995
```

## Run The API

```bash
python pipeline_server.py
```

Default endpoint:

```text
http://127.0.0.1:8002/analyze
```

## Request Format

```json
{
  "request_id": "pipe-001",
  "images": [
    {
      "image_id": "img-001",
      "image_base64": "data:image/jpeg;base64,..."
    }
  ]
}
```

`image_url` is also accepted.

## Test Client

```bash
python pipeline_client.py --image ./sample.png
python pipeline_client.py --image-url https://example.com/sample.jpg
```

## Docker Compose

`docker-compose.yml` runs:

- `triton`: serves `dedup_embedder`, `AI_images_detector`, and `nfa_vit`
- `pipeline`: FastAPI service that calls Triton, Milvus, and OpenAI

Run:

```bash
docker compose up -d --build
```

Default published ports:

- Triton HTTP: `30000`
- Triton gRPC: `30001`
- Triton Metrics: `30002`
- Pipeline API: `8002`

## Model Repository

Place the actual model files into:

```text
model_repository/AI_images_detector/1/model.onnx
model_repository/dedup_embedder/1/model.onnx
model_repository/nfa_vit/1/nfa_vit.onnx
model_repository/nfa_vit/1/nfa_vit.onnx.data
```

The provided `config.pbtxt` files assume:

- dedup input: `pixel_values`
- dedup output: `features`
- classification input: `pixel_values`
- classification output: `logits`
- nfa input: `image`, `mask`, `label`
- nfa output: `pred_mask`, `pred_label`

Adjust the config or env if your ONNX signatures differ.

## Migration Note

The Milvus cache schema now stores `nfa_vit` output and `final_label`.
If you already have an existing collection created by an older version of the pipeline, delete that collection before restarting so the API can recreate it with the new fields.
