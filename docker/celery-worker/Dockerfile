# ─────────────────────────────────────────────────────────────────────────────
# BookHead AI – Dockerfile - Celery Worker Service.
# ─────────────────────────────────────────────────────────────────────────────

## To build docker image move to project root directory - ..\book-shelves-ai-app
## Run this command: 
##          docker build -f "docker celery-worker"/Dockerfile -t bookhead-worker:<version-tag> .

## This creates Docker Image for Celery Worker Service

# ─────────────────────────────────────────────
# STAGE 1 — Builder
# ─────────────────────────────────────────────
FROM python:3.11 AS builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    pkg-config \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# NOTE: libgomp1 -> llama-cpp-python depends on the GNU OpenMP runtime library (libgomp.so.1).

WORKDIR /app

# Upgrade pip tooling
RUN python -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    cmake \
    ninja

# llama.cpp build settings
ENV CMAKE_ARGS="-DLLAMA_NATIVE=OFF"
ENV FORCE_CMAKE=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=1

# Copy requirements first for layer caching 
COPY requirements-worker-service.txt .

# Install dependencies into custom folder -------------------------------------------------------
RUN pip install --prefix=/install llama-cpp-python==0.2.90
RUN pip install --prefix=/install --no-cache-dir -r requirements-worker-service.txt

## Ensuring the correct versions of problematic packages:
RUN pip install --prefix=/install --force-reinstall numpy==1.26.4
RUN pip install --prefix=/install --force-reinstall packaging==26.2

    
# ─────────────────────────────────────────────
# STAGE 2 — Runtime
# ─────────────────────────────────────────────
FROM python:3.11-slim

# ONLY runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgomp1
# NOTE: libgomp1 -> llama-cpp-python depends on the GNU OpenMP runtime library (libgomp.so.1).
#                   Minimal Docker images (i.e., python:slim) do not include it by default

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source - {copy this from current project directory in host} TO {this path in Docker Image}
#                                                                  . -> means copy inside WORKDIR /app
#                                                                  ./app -> means copy inside /app/app
COPY app ./app
COPY main.py .
COPY frontend ./frontend
COPY requirements-worker-service.txt .


# Create uploads & models dir; later they will be volume mounted during runtime
RUN mkdir -p /app/uploads
RUN mkdir -p /app/models

# Runtime env
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Default: Start Celery Worker with thread concurrency of 2
CMD ["python", "-m", "celery", "-A", "app.core.celery_app", "worker", "--loglevel=info", "--pool=threads", "--concurrency=2"]


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────