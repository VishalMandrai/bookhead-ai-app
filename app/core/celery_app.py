# ─────────────────────────────────────────────────────────────────────────────
# app/core/celery_app.py
#
# Celery application factory.
#
# Why Celery?
#   ML inference (detection + OCR) is slow – often 5–30 seconds per image.
#   Blocking the FastAPI request thread that long would time out HTTP clients
#   and waste server resources. Instead, we:
#     1. Accept the upload via FastAPI (fast – just saves the file).
#     2. Dispatch a Celery task (near-instant – just enqueues a message).
#     3. Return a `job_id` to the client immediately (202 Accepted).
#     4. Let the Celery worker process the ML pipeline in the background.
#     5. Client polls GET /jobs/{job_id} until the result is ready.
#
# Redis serves as both the message broker (task queue) and the result backend
# (stores task output so the API can retrieve it on poll).
# ─────────────────────────────────────────────────────────────────────────────

from celery import Celery
from app.core.config import get_settings
from celery.signals import worker_process_init
import logging

logger = logging.getLogger(__name__)


def create_celery_app() -> Celery:
    """
    Creates and configures the Celery application instance.

    Keeping this in a factory function (rather than module-level) makes it
    easy to swap configuration in tests without side effects.
    """
    settings = get_settings()

    celery_app = Celery(
        "booklens",                         # Application name
        broker=settings.redis_url,          # Where tasks are enqueued
        backend=settings.redis_url,         # Where results are stored
        include=[                           # Task modules to auto-discover
            "app.pipelines.reader_pipeline",
            "app.pipelines.librarian_pipeline",
        ],
    )

    celery_app.conf.update(
        # ── Serialisation ─────────────────────────────────────────────────────
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],

        # ── Reliability ───────────────────────────────────────────────────────
        task_acks_late=True,            # Only ACK a task after it completes
        task_reject_on_worker_lost=True,  # Re-queue if worker crashes mid-task
        task_track_started=True,        # Allows "STARTED" state in polling

        # ── Result expiry ─────────────────────────────────────────────────────
        # Keep results in Redis for 1 hour; long enough for any UI poll cycle
        result_expires=3600,

        # ── Routing ───────────────────────────────────────────────────────────
        # All tasks use the default queue for now.
        # Future: separate "heavy" queue for detection tasks with GPU workers.
        task_default_queue="default",
    )

    return celery_app


# Module-level singleton used by task decorators and the worker process
celery_app = create_celery_app()


# Prime per-process cached services when a worker process starts. This ensures
# heavy ML models are loaded once per worker process (not per task) and are
# available for all tasks executed by that process.
@worker_process_init.connect
def _prime_worker_services(**_kwargs):
    try:
        # Import inside the handler to avoid importing heavy modules at
        # Celery app creation time; `get_worker_services` only instantiates
        # models when called.
        from app.core.worker_services import get_worker_services
        
        logger.info("starting_up_worker_services")

        get_worker_services()
        logger.info("worker_services_warmed")
    except Exception as exc:
        # Use logging here; failing to warm is non-fatal (tasks will still
        # build lazily on first use), but we surface the error for ops.
        logger.warning("worker_services_warm_failed", exc=exc)


## NOTE: "worker_process_init" ensures:
##      1. Models are initialized once per worker process
##      2. Reused across all tasks handled by that process

## NOTE: Why we need "worker_process_init" - 
## We have:
##    - 4–5 ML models (large, expensive to load)
##    - Repeated task execution via Celery
##    - No need for these models in FastAPI

## So:
##    - Loading inside FastAPI → wasteful ❌
##    - Loading per task → disastrous latency ❌
##    - Loading once per worker process → CORRECT pattern ✅ (STILL 1 BIG PROBLEM)

## BIG PROBLEM: 
## If we run -  celery -A app.core.clerey_app worker --concurrency=4

# Then:
    # - we get 4 separate OS processes
    # - Each process runs worker_process_init
    # - Each process loads its own copy of your models
    # 👉 That means:
            # If 1 model = 500 MB
            # → 5 models = 2.5 GB
            # → 4 workers = 10 GB RAM

## So the design is CORRECT logically, BUT can become very expensive physically.


## NOTE SOLUTION:       celery -A app worker --pool=threads --concurrency=2      (NOTE: This configuration is best for Gemma 2B LLM; as it internally spawns out 4 threads within a worker thread for inference. Hence best balance for 8 core CPU.)
# Now:
        # - Single process; 1 Worker Process with 2 Threads each working concurrently
        # - Shared memory
        # - Models loaded once
        # - Multiple threads reuse them

## ⚠️ BUT: Python GIL affects CPU-bound workloads

## Works best if:
        # 1. Models use native libs (NumPy, PyTorch, TensorFlow release GIL)
        # 2. Or inference is I/O heavy
        
        
## NOTE ANOTHER SOLUTION: External Model Server (best for scale)

# If models are heavy:
        # - Move them to a separate inference service
        # - Celery calls that service

# Example stack:
        # - FastAPI + Models (inference server)
        # - Celery = orchestration layer