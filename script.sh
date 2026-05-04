#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./script.sh <model.onnx> [model.engine]

Examples:
  ./script.sh dinov3_vits16plus.onnx
  ./script.sh dinov3_vits16plus.onnx dinov3_vits16plus_fp16.engine

Environment variables:
  TRT_DOCKER_IMAGE      TensorRT Docker image. If empty, auto-detects a local image.
  INPUT_NAME            ONNX input tensor name. Default: pixel_values
  MIN_SHAPE             TensorRT min shape. Default: 1x3x224x224
  OPT_SHAPE             TensorRT opt shape. Default: 1x3x224x224
  MAX_SHAPE             TensorRT max shape. Default: 8x3x224x224
  PRECISION             fp32, fp16, or int8. Default: fp16
  WORKSPACE_MB          TensorRT workspace memory in MB. Default: 4096
  SKIP_SHAPES           Set to 1 for a fully static ONNX model.
  EXTRA_TRTEXEC_ARGS    Extra trtexec args, for example: "--verbose --dumpLayerInfo"

Notes:
  This script expects Docker GPU support and a TensorRT container with trtexec.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

abs_existing_file() {
  local path="$1"
  local dir
  local base

  [ -f "$path" ] || die "ONNX file not found: $path"
  dir="$(dirname "$path")"
  base="$(basename "$path")"
  (cd "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base")
}

abs_target_file() {
  local path="$1"
  local dir
  local base

  dir="$(dirname "$path")"
  base="$(basename "$path")"
  mkdir -p "$dir"
  (cd "$dir" && printf '%s/%s\n' "$(pwd -P)" "$base")
}

detect_tensorrt_image() {
  docker images --format '{{.Repository}}:{{.Tag}}' \
    | awk '
      /(^|\/)tensorrt:/ || /nvcr\.io\/nvidia\/tensorrt:/ || /nvidia\/tensorrt:/ {
        if ($0 !~ /:<none>$/) {
          print
          exit
        }
      }
    '
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

[ "$#" -ge 1 ] || {
  usage
  exit 1
}

require_command docker

ONNX_PATH="$(abs_existing_file "$1")"
DEFAULT_ENGINE_PATH="${ONNX_PATH%.*}.engine"
ENGINE_PATH="$(abs_target_file "${2:-$DEFAULT_ENGINE_PATH}")"

ONNX_DIR="$(dirname "$ONNX_PATH")"
ONNX_FILE="$(basename "$ONNX_PATH")"
ENGINE_DIR="$(dirname "$ENGINE_PATH")"
ENGINE_FILE="$(basename "$ENGINE_PATH")"

TRT_DOCKER_IMAGE="${TRT_DOCKER_IMAGE:-$(detect_tensorrt_image)}"
[ -n "$TRT_DOCKER_IMAGE" ] || die "Cannot auto-detect TensorRT Docker image. Set TRT_DOCKER_IMAGE."

INPUT_NAME="${INPUT_NAME:-pixel_values}"
MIN_SHAPE="${MIN_SHAPE:-1x3x224x224}"
OPT_SHAPE="${OPT_SHAPE:-1x3x224x224}"
MAX_SHAPE="${MAX_SHAPE:-8x3x224x224}"
PRECISION="${PRECISION:-fp16}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
SKIP_SHAPES="${SKIP_SHAPES:-0}"
EXTRA_TRTEXEC_ARGS="${EXTRA_TRTEXEC_ARGS:-}"

case "$PRECISION" in
  fp32|fp16|int8) ;;
  *) die "PRECISION must be one of: fp32, fp16, int8" ;;
esac

echo "TensorRT image : $TRT_DOCKER_IMAGE"
echo "ONNX input     : $ONNX_PATH"
echo "Engine output  : $ENGINE_PATH"
echo "Input profile  : ${INPUT_NAME} min=${MIN_SHAPE} opt=${OPT_SHAPE} max=${MAX_SHAPE}"
echo "Precision      : $PRECISION"

docker run --rm --gpus all \
  -e TRTEXEC_BIN="${TRTEXEC_BIN:-trtexec}" \
  -e CONTAINER_ONNX="/workspace/input/${ONNX_FILE}" \
  -e CONTAINER_ENGINE="/workspace/output/${ENGINE_FILE}" \
  -e INPUT_NAME="$INPUT_NAME" \
  -e MIN_SHAPE="$MIN_SHAPE" \
  -e OPT_SHAPE="$OPT_SHAPE" \
  -e MAX_SHAPE="$MAX_SHAPE" \
  -e PRECISION="$PRECISION" \
  -e WORKSPACE_MB="$WORKSPACE_MB" \
  -e SKIP_SHAPES="$SKIP_SHAPES" \
  -e EXTRA_TRTEXEC_ARGS="$EXTRA_TRTEXEC_ARGS" \
  -v "${ONNX_DIR}:/workspace/input:ro" \
  -v "${ENGINE_DIR}:/workspace/output" \
  "$TRT_DOCKER_IMAGE" \
  bash -lc '
    set -Eeuo pipefail

    if command -v "$TRTEXEC_BIN" >/dev/null 2>&1; then
      TRTEXEC="$(command -v "$TRTEXEC_BIN")"
    elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
      TRTEXEC="/usr/src/tensorrt/bin/trtexec"
    else
      echo "ERROR: trtexec not found in TensorRT container." >&2
      exit 127
    fi

    args=(
      --onnx="$CONTAINER_ONNX"
      --saveEngine="$CONTAINER_ENGINE"
      --memPoolSize="workspace:${WORKSPACE_MB}"
    )

    if [ "$SKIP_SHAPES" != "1" ]; then
      args+=(
        --minShapes="${INPUT_NAME}:${MIN_SHAPE}"
        --optShapes="${INPUT_NAME}:${OPT_SHAPE}"
        --maxShapes="${INPUT_NAME}:${MAX_SHAPE}"
      )
    fi

    case "$PRECISION" in
      fp16) args+=(--fp16) ;;
      int8) args+=(--int8) ;;
      fp32) ;;
    esac

    if [ -n "$EXTRA_TRTEXEC_ARGS" ]; then
      # shellcheck disable=SC2206
      extra_args=($EXTRA_TRTEXEC_ARGS)
      args+=("${extra_args[@]}")
    fi

    echo "Running: $TRTEXEC ${args[*]}"
    "$TRTEXEC" "${args[@]}"
  '

[ -s "$ENGINE_PATH" ] || die "TensorRT engine was not created: $ENGINE_PATH"
echo "Done: $ENGINE_PATH"
