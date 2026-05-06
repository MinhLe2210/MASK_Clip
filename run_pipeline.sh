#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_pipeline.sh

Optional environment:
  PYTHON_BIN             Python executable. Default: python
  HOST                   Bind host for all services. Default: 127.0.0.1
  DEDUP_PORT             Dedup API port. Default: 8000
  CLASSIFIER_PORT        Classification API port. Default: 8001
  PIPELINE_PORT          Pipeline API port. Default: 8002
  STARTUP_TIMEOUT        Seconds to wait per service. Default: 120

Required environment:
  DEDUP_ENGINE_PATH      TensorRT engine for dedup embeddings
  CLASSIFIER_ENGINE_PATH TensorRT engine for classification
  MILVUS_HOST            Milvus host
  OPENAI_API_KEY         OpenAI API key for /analyze

The script auto-loads .env from the repo root when present.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: Missing command: $1" >&2
    exit 1
  }
}

require_env() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "ERROR: Missing environment variable: $name" >&2
    exit 1
  fi
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local timeout="$3"

  "$PYTHON_BIN" - "$name" "$url" "$timeout" <<'PY'
import sys
import time
import urllib.error
import urllib.request

name, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.time() + timeout
last_error = "service did not answer"

while time.time() < deadline:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2):
            sys.exit(0)
    except urllib.error.HTTPError as exc:
        if 100 <= exc.code < 500:
            sys.exit(0)
        last_error = f"HTTP {exc.code}"
    except Exception as exc:
        last_error = str(exc)
    time.sleep(1)

print(f"{name} is not ready at {url}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

show_log_tail() {
  local title="$1"
  local file="$2"
  if [ -f "$file" ]; then
    echo "----- ${title} (${file}) -----" >&2
    tail -n 80 "$file" >&2 || true
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  for pid in "${PIPELINE_PID:-}" "${CLASSIFIER_PID:-}" "${DEDUP_PID:-}"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done

  wait || true
  exit "$status"
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${HOST:-127.0.0.1}"
DEDUP_PORT="${DEDUP_PORT:-8000}"
CLASSIFIER_PORT="${CLASSIFIER_PORT:-8001}"
PIPELINE_PORT="${PIPELINE_PORT:-8002}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-120}"

export HOST
export MODEL_PATH="${MODEL_PATH:-${DEDUP_ENGINE_PATH:-}}"
export DEDUP_ENGINE_PATH="${DEDUP_ENGINE_PATH:-${MODEL_PATH:-}}"
export EMBED_DIM="${EMBED_DIM:-384}"
export CLASSIFIER_INPUT_NAME="${CLASSIFIER_INPUT_NAME:-input}"
export CLASSIFIER_OUTPUT_NAME="${CLASSIFIER_OUTPUT_NAME:-output}"
export CLASSIFIER_LABELS="${CLASSIFIER_LABELS:-negative,positive}"
export CLASSIFIER_INPUT_SIZE="${CLASSIFIER_INPUT_SIZE:-384}"
export OPENAI_VLM_MODEL="${OPENAI_VLM_MODEL:-gpt-5-mini}"
export OPENAI_PROXY="${OPENAI_PROXY:-}"
export DEDUP_API_URL="${DEDUP_API_URL:-http://${HOST}:${DEDUP_PORT}/dedup}"
export CLASSIFIER_API_URL="${CLASSIFIER_API_URL:-http://${HOST}:${CLASSIFIER_PORT}/classify}"

require_command "$PYTHON_BIN"
require_env DEDUP_ENGINE_PATH
require_env CLASSIFIER_ENGINE_PATH
require_env MILVUS_HOST
require_env OPENAI_API_KEY

[ -f "$DEDUP_ENGINE_PATH" ] || {
  echo "ERROR: DEDUP_ENGINE_PATH does not exist: $DEDUP_ENGINE_PATH" >&2
  exit 1
}

[ -f "$CLASSIFIER_ENGINE_PATH" ] || {
  echo "ERROR: CLASSIFIER_ENGINE_PATH does not exist: $CLASSIFIER_ENGINE_PATH" >&2
  exit 1
}

RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/opensdi-pipeline.XXXXXX")"
DEDUP_LOG="$RUNTIME_DIR/dedup.log"
CLASSIFIER_LOG="$RUNTIME_DIR/classification.log"
PIPELINE_LOG="$RUNTIME_DIR/pipeline.log"

trap cleanup EXIT INT TERM

echo "Runtime dir: $RUNTIME_DIR"
echo "Starting dedup service on ${HOST}:${DEDUP_PORT}"
(
  export PORT="$DEDUP_PORT"
  exec "$PYTHON_BIN" main.py
) >"$DEDUP_LOG" 2>&1 &
DEDUP_PID=$!

if ! wait_for_http "dedup" "$DEDUP_API_URL" "$STARTUP_TIMEOUT"; then
  show_log_tail "dedup" "$DEDUP_LOG"
  exit 1
fi

echo "Starting classification service on ${HOST}:${CLASSIFIER_PORT}"
(
  export CLASSIFIER_PORT
  exec "$PYTHON_BIN" classification_server.py
) >"$CLASSIFIER_LOG" 2>&1 &
CLASSIFIER_PID=$!

if ! wait_for_http "classification" "$CLASSIFIER_API_URL" "$STARTUP_TIMEOUT"; then
  show_log_tail "classification" "$CLASSIFIER_LOG"
  exit 1
fi

echo "Starting pipeline service on ${HOST}:${PIPELINE_PORT}"
(
  export PIPELINE_PORT
  exec "$PYTHON_BIN" pipeline_server.py
) >"$PIPELINE_LOG" 2>&1 &
PIPELINE_PID=$!

PIPELINE_URL="http://${HOST}:${PIPELINE_PORT}/analyze"
if ! wait_for_http "pipeline" "$PIPELINE_URL" "$STARTUP_TIMEOUT"; then
  show_log_tail "pipeline" "$PIPELINE_LOG"
  exit 1
fi

cat <<EOF
Pipeline is ready.
  dedup          $DEDUP_API_URL
  classification $CLASSIFIER_API_URL
  pipeline       $PIPELINE_URL

Logs:
  $DEDUP_LOG
  $CLASSIFIER_LOG
  $PIPELINE_LOG
EOF

wait -n "$DEDUP_PID" "$CLASSIFIER_PID" "$PIPELINE_PID" || true

echo "A service exited. Recent logs:" >&2
show_log_tail "dedup" "$DEDUP_LOG"
show_log_tail "classification" "$CLASSIFIER_LOG"
show_log_tail "pipeline" "$PIPELINE_LOG"
exit 1
