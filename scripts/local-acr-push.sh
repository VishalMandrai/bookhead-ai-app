#!/usr/bin/env bash
set -euo pipefail

ACR_LOGIN_SERVER="${ACR_LOGIN_SERVER:?Please set ACR_LOGIN_SERVER}"
ACR_USERNAME="${ACR_USERNAME:?Please set ACR_USERNAME}"
ACR_PASSWORD="${ACR_PASSWORD:?Please set ACR_PASSWORD}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "Logging into Azure Container Registry: $ACR_LOGIN_SERVER"
docker login "$ACR_LOGIN_SERVER" -u "$ACR_USERNAME" -p "$ACR_PASSWORD"

echo "Building FastAPI image"
docker build -t "$ACR_LOGIN_SERVER/bookhead-fastapi:$IMAGE_TAG" -f docker/fastapi/Dockerfile .

echo "Building Celery worker image"
docker build -t "$ACR_LOGIN_SERVER/bookhead-worker:$IMAGE_TAG" -f docker/celery-worker/Dockerfile .

echo "Pushing images to ACR"
docker push "$ACR_LOGIN_SERVER/bookhead-fastapi:$IMAGE_TAG"
docker push "$ACR_LOGIN_SERVER/bookhead-worker:$IMAGE_TAG"

echo "Pushed images:"
echo "  $ACR_LOGIN_SERVER/bookhead-fastapi:$IMAGE_TAG"
echo "  $ACR_LOGIN_SERVER/bookhead-worker:$IMAGE_TAG"
