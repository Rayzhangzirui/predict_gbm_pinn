#!/usr/bin/env bash
# Build and push PINNGBM GPU image (linux/amd64) to GHCR.
# Optional: also export a local tarball (mainly for archiving).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE_NAME="${IMAGE_NAME:-pinngbm}"
TAG="${IMAGE_TAG:-gpu-latest}"                 # use gpu-latest by default
GHCR_USER="${GHCR_USER:-rayzhangzirui}"
PLATFORM="${PLATFORM:-linux/amd64}"

GHCR_REF="ghcr.io/${GHCR_USER}/${IMAGE_NAME}:${TAG}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.gpu}"

# Set EXPORT_TAR=1 if you want a tarball (note: will require a separate local build)
EXPORT_TAR="${EXPORT_TAR:-0}"
TAR_PATH="${TAR_PATH:-${PROJECT_ROOT}/${IMAGE_NAME}_${TAG}.tar}"

log() { echo "[$(date +'%H:%M:%S')] $*"; }
die() { echo "Error: $*" >&2; exit 1; }

log "Project:    $PROJECT_ROOT"
log "Dockerfile: $DOCKERFILE"
log "Image:      $GHCR_REF"
log "Platform:   $PLATFORM"

# --- Login (needed for push) ---
if [[ -z "${GHCR_TOKEN:-}" ]]; then
  die "Set GHCR_TOKEN (GitHub PAT with write:packages). Example: export GHCR_TOKEN=..."
fi
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null
log "Logged into GHCR as $GHCR_USER"

# --- Buildx push (this is the cluster image) ---
docker buildx create --use >/dev/null 2>&1 || true
log "Building and pushing (buildx)..."
docker buildx build \
  --platform "$PLATFORM" \
  -f "$DOCKERFILE" \
  -t "$GHCR_REF" \
  --push \
  .

log "Pushed: $GHCR_REF"
echo ""
echo "Cluster usage:"
echo "  singularity build ${IMAGE_NAME}.sif docker://${GHCR_REF}"
echo "  singularity run --nv ${IMAGE_NAME}.sif [args...]"
echo ""