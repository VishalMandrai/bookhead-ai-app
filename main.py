# ─────────────────────────────────────────────────────────────────────────────
# app/main.py
#
# FastAPI application factory.
#
# This file is the entry point for the ASGI server (uvicorn). It:
#   1. Configures logging
#   2. Creates the FastAPI app instance with metadata
#   3. Registers lifespan events (startup / shutdown hooks)
#   4. Mounts routers and static files
#   5. Registers global exception handlers
#
# SOLID note (Single Responsibility):
#   This file only assembles the application. Business logic, service
#   instantiation, and route handling live in their respective modules.
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.exceptions import (
    BookLensError,
    FileTooLargeError,
    JobNotFoundError,
    JobNotReadyError,
    NoBooksDetectedError,
    ReviewSessionNotFoundError,
    ReviewAlreadyCompleteError,
    UnsupportedFileTypeError,
)
from app.api.routes import reader, librarian

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: code that runs on startup and shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for application lifespan events.

    Everything BEFORE `yield` runs at startup.
    Everything AFTER `yield` runs at shutdown.

    Using the lifespan pattern (vs deprecated @app.on_event) is the modern
    FastAPI approach and plays nicely with testing.
    """
    settings = get_settings()

    # ── Startup ────────────────────────────────────────────────────────────
    configure_logging()
    logger.info("booklens_starting", env=settings.env, version="0.1.0")

    # Ensure the uploads directory exists on every startup
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    logger.info("uploads_dir_ready", path=settings.upload_dir)

    ## Warm up ML models so the first request doesn't pay model-loading cost
    # Only in "PRODUCTION" to keep startup during the devlopement fast
    # if settings.load_models_on_startup:
    #     from app.api.dependencies import get_detection_service, get_ocr_service
    #     get_detection_service()
    #     get_ocr_service()
        
    logger.info("booklens_ready")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("booklens_shutting_down")


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.

    Using a factory function (rather than module-level `app = FastAPI()`)
    makes it trivial to create fresh app instances in tests without shared
    global state.
    """
    settings = get_settings()

    app = FastAPI(
        title="BookLens AI",
        description=(
            "An AI-powered web app for book readers and librarians. "
            "Upload shelf images to get personalised reading recommendations "
            "or generate a structured library catalog."
        ),
        version="0.1.0",
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    # Permissive in development; lock down origins in production via env var.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(reader.router)     # Reader routes are under /reader/*
    app.include_router(librarian.router)  # Librarian routes are under /librarian/*


    # ── Health check ──────────────────────────────────────────────────────
    @app.get("/health", tags=["System"], include_in_schema=True)
    async def health():
        """Liveness probe for load balancers and Docker health checks."""
        return {"status": "ok", "version": "0.1.0"}
    
    
    # ── Static files ──────────────────────────────────────────────────────
    # Serve uploaded and cropped images at /uploads/<filename>
    # The review UI fetches crop images from this path.
    uploads_path = Path(settings.upload_dir)
    uploads_path.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")

    # Serve the frontend HTML/CSS/JS at the root
    frontend_path = Path("frontend/static")
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


    # ── Root route: serve the frontend SPA ──────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def serve_root():
        """Serve the main frontend HTML from frontend/templates/index.html."""
        index_path = Path("frontend/templates/index.html")
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse({"message": "BookLens AI is running. See /docs for the API."})



    # ── Exception handlers ────────────────────────────────────────────────
    _register_exception_handlers(app)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """
    Maps custom exceptions to appropriate HTTP status codes.

    This is the single place that decides "which exception class → which HTTP
    code". Route handlers simply raise the domain exception; they never
    construct HTTP error responses themselves.
    """

    @app.exception_handler(JobNotFoundError)
    async def job_not_found_handler(request: Request, exc: JobNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "job_not_found", "message": exc.message, **exc.detail},
        )

    @app.exception_handler(JobNotReadyError)
    async def job_not_ready_handler(request: Request, exc: JobNotReadyError):
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"error": "job_not_ready", "message": exc.message},
        )

    @app.exception_handler(ReviewSessionNotFoundError)
    async def review_not_found_handler(request: Request, exc: ReviewSessionNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "review_session_not_found", "message": exc.message},
        )

    @app.exception_handler(ReviewAlreadyCompleteError)
    async def review_already_complete_handler(request: Request, exc: ReviewAlreadyCompleteError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": "review_already_complete", "message": exc.message},
        )

    @app.exception_handler(NoBooksDetectedError)
    async def no_books_handler(request: Request, exc: NoBooksDetectedError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "no_books_detected", "message": exc.message},
        )

    @app.exception_handler(FileTooLargeError)
    async def file_too_large_handler(request: Request, exc: FileTooLargeError):
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": "file_too_large", "message": exc.message},
        )

    @app.exception_handler(BookLensError)
    async def generic_booklens_handler(request: Request, exc: BookLensError):
        """Catch-all for any unhandled BookLensError subclass → 500."""
        logger.error("unhandled_booklens_error", error=str(exc), detail=exc.detail)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_error", "message": exc.message},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level app instance (used by uvicorn: `uvicorn app.main:app`)
# ─────────────────────────────────────────────────────────────────────────────
app = create_app()
