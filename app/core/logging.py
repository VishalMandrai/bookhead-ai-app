# ─────────────────────────────────────────────────────────────────────────────
# app/core/logging.py
#
# Configures structlog for structured, JSON-formatted logging.
#
# Structured logging means every log entry is a machine-readable JSON object
# (timestamp, level, logger name, message, and any extra key-value context).
# This is far easier to query in production log aggregators (Datadog, Loki, etc.)
# than plain text.
#
# Usage anywhere in the codebase:
#   from app.core.logging import get_logger
#   logger = get_logger(__name__)
#   logger.info("ocr_complete", book_id=book.id, confidence=0.91)
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import structlog
from app.core.config import get_settings


def configure_logging() -> None:
    """
    Call once at application startup (in main.py lifespan).
    Sets up structlog processors and ties it to the standard-library
    logging backend so third-party libraries (uvicorn, celery, etc.)
    also emit structured output.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ── Standard-library logging ──────────────────────────────────────────────
    # Route all stdlib loggers through structlog's handler so everything
    # (including uvicorn access logs) comes out in the same structured format.
    logging.basicConfig(
        format="%(message)s",           # structlog handles formatting
        stream=sys.stdout,
        level=log_level,
    )

    # ── Shared processors (run on every log event) ────────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,   # Thread/async-local context
        structlog.stdlib.add_logger_name,          # Adds "logger" key
        structlog.stdlib.add_log_level,            # Adds "level" key
        structlog.processors.TimeStamper(fmt="iso"),  # ISO-8601 timestamp
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,      # Pretty exception tracebacks
    ]

    # ── Output format ─────────────────────────────────────────────────────────
    # Development: colourful, human-readable console output.
    # Production:  compact JSON for log aggregators.
    if settings.is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Apply the renderer to the stdlib formatter so uvicorn output is also
    # routed through structlog.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Returns a bound structlog logger for the given module name.

    Example:
        logger = get_logger(__name__)
        logger.info("detection_complete", num_books=5, duration_ms=340)
    """
    return structlog.get_logger(name)
