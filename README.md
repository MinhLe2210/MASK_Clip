# OpenSDI

OpenSDI currently contains two runtime flows:

- `/dedup`: LitServe API for base64 image deduplication with Milvus.
- `/classify`: LitServe API for 2-class TensorRT classification.
- `/analyze`: pipeline API that runs dedup -> TensorRT classification -> OpenAI multimodal analysis.

## Project Layout

```text
.
|-- main.py                    # LitServe wiring for the dedup API
|-- classification_server.py   # LitServe TensorRT classification API
|-- pipeline_server.py         # Dedup -> classify -> OpenAI VLM pipeline
|-- convert_to_onnx.py         # Export DINOv3 image model to ONNX
|-- script.sh                  # Convert ONNX to TensorRT engine with Docker trtexec
|-- pipeline_client.py         # Test client for pipeline_server.py
`-- src/
    |-- config.py              # Environment settings
    |-- dedup_service.py       # Deduplication workflow
    |-- embedder.py            # Hugging Face image embedding model
    |-- image_io.py            # Base64/PIL helpers
    |-- milvus_store.py        # Milvus collection, search, insert, retry
    |-- request_parsing.py     # LitServe request batching/unbatching helpers
    `-- trt_runner.py          # TensorRT engine loader/inference helper
```

## Dedup API

Required environment variables:

```bash
export MODEL_PATH=facebook/dinov3-vits16plus-pretrain-lvd1689m
export MILVUS_HOST=127.0.0.1
export MILVUS_PORT=19530
```

Common optional variables:

```bash
export EMBED_DIM=768
export DUP_THRESHOLD=0.999995
export COLLECTION_NAME=AI_detector_image_dedup_b64
export MAX_IMAGES_PER_REQUEST=64
export PORT=8000
```

Run:

```bash
python main.py
```

Request:

```bash
curl -X POST http://127.0.0.1:8000/dedup \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-001",
    "images": [
      {
        "image_id": "img-001",
        "image_base64": "data:image/jpeg;base64,..."
      }
    ]
  }'
```

Response statuses:

- `unique`: image was accepted and inserted into Milvus.
- `duplicate`: image matched by SHA256 or embedding similarity.
- `error`: invalid image/base64 payload.

## Export ONNX

Export DINOv3 to ONNX:

```bash
python convert_to_onnx.py \
  --model-path facebook/dinov3-vits16plus-pretrain-lvd1689m \
  --image-url http://images.cocodataset.org/val2017/000000039769.jpg \
  --output dinov3_vits16plus.onnx
```

Use raw `pooler_output` instead of normalized embedding:

```bash
python convert_to_onnx.py \
  --image-url http://images.cocodataset.org/val2017/000000039769.jpg \
  --output dinov3_vits16plus_raw.onnx \
  --no-normalize
```

## ONNX To TensorRT

The TensorRT Docker image must already be pulled locally.

```bash
chmod +x script.sh
./script.sh dinov3_vits16plus.onnx dinov3_vits16plus_fp16.engine
```

If auto-detect does not find the image:

```bash
TRT_DOCKER_IMAGE=nvcr.io/nvidia/tensorrt:xx.xx-py3 \
./script.sh dinov3_vits16plus.onnx dinov3_vits16plus_fp16.engine
```

Useful overrides:

```bash
PRECISION=fp32 ./script.sh model.onnx model_fp32.engine
MAX_SHAPE=16x3x224x224 ./script.sh model.onnx model_b16.engine
SKIP_SHAPES=1 ./script.sh static_model.onnx static_model.engine
```

## Classification API

The classification service is a TensorRT `.engine` API. It expects an engine
with one image input tensor and one 2-logit output tensor, for example:

```bash
Input tensor: input, shape=(B, 3, 384, 384)
Output tensor: output, shape=(B, 2)
```

Run:

```bash
export CLASSIFIER_ENGINE_PATH=./classifier.engine
export CLASSIFIER_LABELS=negative,positive
export CLASSIFIER_INPUT_NAME=input
export CLASSIFIER_OUTPUT_NAME=output
export CLASSIFIER_INPUT_SIZE=384
export CLASSIFIER_BATCH_SIZE=16
export CLASSIFIER_PORT=8001
python classification_server.py
```

Request:

```bash
curl -X POST http://127.0.0.1:8001/classify \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "cls-001",
    "images": [
      {
        "image_id": "img-001",
        "image_base64": "data:image/jpeg;base64,..."
      }
    ]
  }'
```

Preprocessing matches the TensorRT example:

```text
RGB -> resize 384x384 BICUBIC -> /255 -> ImageNet mean/std -> NCHW float32
```

## Pipeline API

The pipeline service runs:

```text
dedup API -> if unique, TensorRT classification API -> OpenAI multimodal model
```

Required environment:

```bash
export OPENAI_API_KEY=...
export OPENAI_VLM_MODEL=gpt-5-mini
export DEDUP_API_URL=http://127.0.0.1:8000/dedup
export CLASSIFIER_API_URL=http://127.0.0.1:8001/classify
```

Run:

```bash
python pipeline_server.py
```

Call:

```bash
python pipeline_client.py --image ./sample.jpg
```

Endpoint:

```bash
curl -X POST http://127.0.0.1:8002/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "pipe-001",
    "images": [
      {
        "image_id": "img-001",
        "image_base64": "data:image/jpeg;base64,..."
      }
    ]
  }'
```

## Docker Compose

`docker-compose.yml` runs the services separately:

- `dedup`: 1 CPU, 1 NVIDIA GPU, LitServe batch size 32.
- `classification`: 1 CPU, 1 NVIDIA GPU, TensorRT classification, LitServe batch size 16.
- `pipeline`: 1 CPU, calls dedup, classification, and OpenAI multimodal API.

Build or provide an application image first, then run Compose:

```bash
export OPENSDI_IMAGE=opensdi:latest
export MILVUS_HOST=milvus
export CLASSIFIER_ENGINE_PATH=/models/classifier.engine
export OPENAI_API_KEY=...
docker compose up -d
```

Ports:

- Dedup API: `http://127.0.0.1:8000/dedup`
- Classification API: `http://127.0.0.1:8001/classify`
- Pipeline API: `http://127.0.0.1:8002/analyze`

## Push Image To Harbor

`docker-compose.yml` uses `OPENSDI_IMAGE` for all app services. To push the app
image to Harbor:

```bash
export HARBOR_REGISTRY=harbor.example.com
export HARBOR_PROJECT=opensdi
export IMAGE_TAG=1.0.0
export OPENSDI_IMAGE=$HARBOR_REGISTRY/$HARBOR_PROJECT/opensdi:$IMAGE_TAG

docker login $HARBOR_REGISTRY
docker build -t $OPENSDI_IMAGE .
docker push $OPENSDI_IMAGE
```

You can also use Docker Compose push after the image exists locally:

```bash
export OPENSDI_IMAGE=harbor.example.com/opensdi/opensdi:1.0.0
docker compose push
```

If you want Compose to build and push in one flow, create a local override file:

```yaml
# docker-compose.build.yml
services:
  dedup:
    build: .
  classification:
    build: .
  pipeline:
    build: .
```

Then run:

```bash
export OPENSDI_IMAGE=harbor.example.com/opensdi/opensdi:1.0.0
docker compose -f docker-compose.yml -f docker-compose.build.yml build
docker compose -f docker-compose.yml -f docker-compose.build.yml push
```

In Rancher, deploy workloads with the same image name:

```text
harbor.example.com/opensdi/opensdi:1.0.0
```

For private Harbor projects, add Harbor credentials in Rancher/Kubernetes as a
registry secret, then attach that secret to the workload image pull settings.

## Notes

- `main.py` is intentionally small and keeps only LitServe API/server wiring.
- Dedup business logic lives under `src/`.
- TensorRT runtime needs `tensorrt` and `pycuda` available inside the server environment.
- This repo does not include Milvus or TensorRT Docker images; run those separately.
