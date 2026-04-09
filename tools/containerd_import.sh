#!/bin/bash
#
# containerd_import.sh — Import a local Docker image into containerd (k8s.io namespace)
#
# Kubernetes with containerd uses its own image store separate from Docker.
# This script bridges the two so k8s can use locally built images
# without a registry (imagePullPolicy: Never).
#
# Usage:
#   ./tools/containerd_import.sh
#   ./tools/containerd_import.sh --tag cuda12.9
#   ./tools/containerd_import.sh --name nccl-tests --tag cuda12.9
#
set -euo pipefail

# ──────────────────── defaults ────────────────────
IMAGE_NAME="nccl-tests"
IMAGE_TAG="cuda12.9"
CTR_NAMESPACE="k8s.io"
DRY_RUN=0

# ──────────────────── usage ────────────────────
usage() {
    cat <<'USAGE'
Usage: containerd_import.sh [OPTIONS]

Options:
  --name NAME          Docker image name (default: nccl-tests)
  --tag TAG            Docker image tag  (default: cuda12.9)
  --namespace NS       containerd namespace (default: k8s.io)
  --dry-run            Print commands without running
  -h, --help           Show this help
USAGE
    exit 0
}

# ──────────────────── parse args ────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)       IMAGE_NAME="$2";    shift 2 ;;
        --tag)        IMAGE_TAG="$2";     shift 2 ;;
        --namespace)  CTR_NAMESPACE="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1;          shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
FULL_IMAGE="docker.io/library/${LOCAL_IMAGE}"

echo "======================================================"
echo "  Importing image into containerd"
echo "======================================================"
echo "  Source (Docker):    ${LOCAL_IMAGE}"
echo "  Target (containerd) ${CTR_NAMESPACE}: ${FULL_IMAGE}"
echo "======================================================"
echo ""

# Verify the Docker image exists
if ! docker image inspect "${LOCAL_IMAGE}" &>/dev/null; then
    echo "ERROR: Docker image '${LOCAL_IMAGE}' not found."
    echo "Build it first with: ./tools/docker_build.sh --tag ${IMAGE_TAG}"
    exit 1
fi

# Check if already imported
if sudo ctr -n "${CTR_NAMESPACE}" images check "${FULL_IMAGE}" &>/dev/null; then
    echo "Image already present in containerd (${CTR_NAMESPACE}/${FULL_IMAGE})"
    echo "Re-importing to pick up any changes..."
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "--- DRY RUN ---"
    echo "docker save ${LOCAL_IMAGE} | sudo ctr -n ${CTR_NAMESPACE} images import -"
    exit 0
fi

echo "Exporting from Docker and importing into containerd..."
docker save "${LOCAL_IMAGE}" | sudo ctr -n "${CTR_NAMESPACE}" images import -

echo ""
echo "Verifying..."
sudo ctr -n "${CTR_NAMESPACE}" images ls | grep "${IMAGE_NAME}" || echo "WARNING: image not found after import"

echo ""
echo "Done. Use in k8s with:"
echo "  image: ${FULL_IMAGE}"
echo "  imagePullPolicy: Never"
