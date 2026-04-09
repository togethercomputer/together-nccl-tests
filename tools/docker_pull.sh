#!/bin/bash
set -e

ORG="${IMAGE_ORG:-togethercomputer}"
IMAGE_NAME="training-performance"
TAG="${IMAGE_TAG:-nccl-tests-ub22.04-cuda12.9-v0.1}"

IMAGE="${ORG}/${IMAGE_NAME}:${TAG}"

echo "Pulling ${IMAGE} ..."
docker pull "${IMAGE}"

echo "Done: ${IMAGE}"
