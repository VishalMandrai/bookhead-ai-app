# ─────────────────────────────────────────────────────────────────────────────
# BookHead AI – Dockerfile - FastAPI App Service
# ─────────────────────────────────────────────────────────────────────────────

## To build docker image move to project root directory - ..\book-shelves-ai-app
## Run this command: 
##          docker build -f "docker fastapi"/Dockerfile -t bookhead-fastapi:<version-tag> .

## This creates Docker Image for FastAPI App Service

FROM python:3.11-slim    
# Note: slim images sometimes crash native builds
# especially heavy C++ compilation

# Install OS-level dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1


# Copy and install Python dependencies first (layer-cached unless requirements change)
COPY requirements-fastapi-service.txt .
RUN pip install --no-cache-dir -r requirements-fastapi-service.txt

# Copy application source {ignore files written in .dockerignore}
# {copy this from current project directory in host} TO {this path in Docker Image}
#                                                        . -> means copy inside WORKDIR /app
#                                                        ./app -> means copy inside /app/app
COPY app ./app
COPY main.py .
COPY frontend ./frontend


# Create the uploads & models directory that will be volume-mounted at runtime
RUN mkdir -p /app/uploads
RUN mkdir -p /app/models

# Default command (overridden per-service in docker-compose.yml)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
