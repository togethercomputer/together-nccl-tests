#!/bin/bash
set -e

ORG="${IMAGE_ORG:-togethercomputer}"
IMAGE_NAME="training-performance"
TAG="${IMAGE_TAG:-nccl-tests-ub22.04-cuda12.9-v0.1}"
LOCAL_IMAGE="${LOCAL_IMAGE:-nccl-tests:cuda12.9}"

IMAGE="${ORG}/${IMAGE_NAME}:${TAG}"

docker tag "${LOCAL_IMAGE}" "${IMAGE}"

echo "Pushing ${IMAGE} ..."
docker push "${IMAGE}"

echo "Done: ${IMAGE}"
