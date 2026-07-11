#!/bin/bash
#
# docker_build.sh — Build the nccl-tests Docker image
#
# Usage:
#   ./tools/docker_build.sh
#   ./tools/docker_build.sh --cuda 12.9.0 --tag my-tag
#   ./tools/docker_build.sh --no-cache
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ──────────────────── defaults ────────────────────
CUDA_VERSION="12.9.0"
UBUNTU_VERSION="22.04"
NCCL_TESTS_VERSION="v2.13.11"
IMAGE_NAME="nccl-tests"
IMAGE_TAG="cuda${CUDA_VERSION%.*}"   # e.g. cuda12.9
NO_CACHE=""
DRY_RUN=0

# ──────────────────── usage ────────────────────
usage() {
    cat <<'USAGE'
Usage: docker_build.sh [OPTIONS]

Options:
  --cuda VERSION       CUDA version (default: 12.9.0)
  --ubuntu VERSION     Ubuntu version (default: 22.04)
  --nccl-tests VER     nccl-tests git tag (default: v2.13.11)
  --name NAME          Image name (default: nccl-tests)
  --tag TAG            Image tag (default: cuda<major.minor>)
  --no-cache           Build without Docker cache
  --dry-run            Print docker build command without running
  -h, --help           Show this help
USAGE
    exit 0
}

# ──────────────────── parse args ────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)          CUDA_VERSION="$2";       shift 2 ;;
        --ubuntu)        UBUNTU_VERSION="$2";     shift 2 ;;
        --nccl-tests)    NCCL_TESTS_VERSION="$2"; shift 2 ;;
        --name)          IMAGE_NAME="$2";         shift 2 ;;
        --tag)           IMAGE_TAG="$2";          shift 2 ;;
        --no-cache)      NO_CACHE="--no-cache";   shift ;;
        --dry-run)       DRY_RUN=1;               shift ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

BUILD_CMD="docker build \
    --build-arg CUDA_VERSION=${CUDA_VERSION} \
    --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
    --build-arg NCCL_TESTS_VERSION=${NCCL_TESTS_VERSION} \
    ${NO_CACHE} \
    -t ${FULL_IMAGE} \
    ${REPO_ROOT}"

echo "======================================================"
echo "  Building nccl-tests Docker image"
echo "======================================================"
echo "  Image:        ${FULL_IMAGE}"
echo "  CUDA:         ${CUDA_VERSION}"
echo "  Ubuntu:       ${UBUNTU_VERSION}"
echo "  nccl-tests:   ${NCCL_TESTS_VERSION}"
echo "  Dockerfile:   ${REPO_ROOT}/Dockerfile"
echo "======================================================"

if [[ $DRY_RUN -eq 1 ]]; then
    echo ""
    echo "--- DRY RUN ---"
    echo "$BUILD_CMD"
    exit 0
fi

echo ""
$BUILD_CMD

echo ""
echo "Built: ${FULL_IMAGE}"
echo "Size:  $(docker image inspect "${FULL_IMAGE}" --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo 'unknown')"
