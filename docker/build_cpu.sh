#!/bin/bash
# Build CPU-only Docker image for PINNGBM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="pinngbm_cpu"
IMAGE_TAG="latest"
FULL_TAG="${IMAGE_NAME}:${IMAGE_TAG}"

PREDICTGBM_MODEL_DIR="${PREDICTGBM_MODEL_DIR:-PredictGBM/predict_gbm/data/models}"
if [[ "$PREDICTGBM_MODEL_DIR" != /* ]]; then
    MODEL_DIR="${PROJECT_ROOT}/${PREDICTGBM_MODEL_DIR}"
else
    MODEL_DIR="${PREDICTGBM_MODEL_DIR}"
fi
OUTPUT_FILE="${MODEL_DIR}/${IMAGE_NAME}.tar"

cd "$PROJECT_ROOT"
docker build -t ${FULL_TAG} -f docker/Dockerfile.cpu .

mkdir -p "${MODEL_DIR}"
docker save -o "${OUTPUT_FILE}" ${FULL_TAG}

echo "✓ Image built and saved to: ${OUTPUT_FILE}"
echo "Use algorithm='${IMAGE_NAME}' in your prediction calls"
