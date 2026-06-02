# ─────────────────────────────────────────────────────────────────────────────
# app/api/dependencies.py
#
# FastAPI Dependency Injection container.
#
# This is the ONLY place in the codebase where concrete service classes are
# imported and instantiated. Route handlers and pipelines receive abstract
# base-class instances injected by FastAPI's `Depends()` mechanism.
#
# WHY THIS MATTERS (SOLID – Dependency Inversion Principle):
#   Route handlers declare what they need (e.g. `BaseOCRService`) without
#   knowing HOW it is implemented. Swapping EasyOCR for Tesseract only
#   requires changing one line here – zero changes to routes or pipelines.
#
# HOW IT WORKS:
#   FastAPI calls the dependency function on each request. The @lru_cache
#   decoration (on service factories) ensures the ML models are loaded ONCE
#   at startup and reused, rather than reloaded per request.
#
# TESTING:
#   In tests, use FastAPI's `app.dependency_overrides` dict to swap in
#   lightweight stub implementations:
#     app.dependency_overrides[get_ocr_service] = lambda: StubOCRService()
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache
from fastapi import Depends

## NOTE: How Depends works:
# - When FastAPI sees Depends(get_ocr_service) in a Route Handler, it knows
#   to call get_ocr_service() to get the OCR service instance for that request.
# - The first time get_ocr_service() is called, it executes and creates the
#   EasyOCRService instance, which is then cached by @lru_cache.
# - On subsequent calls to get_ocr_service(), @lru_cache returns the cached
#   instance, so the same EasyOCRService is reused across requests without
#   reloading the model.
# - This allows for "lazy loading" of services – they are created on first use,
#   not necessarily at app startup, but they are only created once per worker process.

# NOTE: Depends() only works when FastAPI's dependency injection system handles the call. 
# BUT When you call a function (that uses Depends like get_ocr_service) directly in startup code, 
# FastAPI's injection system isn't involved. Depends works fine when function is called from route handlers.


from app.core.config import Settings, get_settings

# ── Import abstract interfaces ─────────────────────────────────────────────
from app.services.base import (
    BaseDetectionService,
    BaseOCRService,
    BaseConfidenceRouter,
    BaseReviewQueueService,
    BaseResultMerger,
    BaseVectorStoreService,
    BaseCatalogService,
    BaseRecommendationService,
)

# ── Import concrete implementations ───────────────────────────────────────
# NOTE: These are the ONLY concrete imports in the entire API layer.
# All other modules import from `base.py` only.

from app.services.detection import FasterRCNNDetectionService
from app.services.ocr import EasyOCRService
from app.services.confidence_router import ThresholdConfidenceRouter
from app.services.review_queue import RedisReviewQueueService
from app.services.result_merger import DefaultResultMerger
from app.services.vector_store import QdrantVectorStoreService
from app.services.catalog import LLMCatalogService
from app.services.recommendation import LLMRecommendationService


# ─────────────────────────────────────────────────────────────────────────────
# Service factories
# Each returns a concrete instance typed as its abstract base class.
# @lru_cache ensures singletons (model loaded once per worker process).
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_detection_service(
    settings: Settings = Depends(get_settings),
) -> BaseDetectionService:
    """
    Provides the book spine detection service.
    Faster-RCNN model weights are loaded here at first call.
    """
    return FasterRCNNDetectionService(
        model_weights_path=settings.detection_model_weights or None,
        upload_dir=settings.upload_dir,
        detection_threshold=settings.detection_threshold,
        nms_iou_threshold=settings.detection_nms_iou_threshold,
        device=settings.detection_device,
    )

@lru_cache(maxsize=1)
def get_ocr_service(
    settings: Settings = Depends(get_settings),
) -> BaseOCRService:
    """
    Provides the OCR service.
    EasyOCR downloads language models on first use; this ensures that
    happens at startup, not mid-request.
    """
    from app.services.ocr_preprocessing import OCRSpinePreprocessor
    from app.services.text_parser import SpineTextParser

    preprocessor = OCRSpinePreprocessor(
        upscale=settings.ocr_preprocess_upscale,
        fix_orientation=settings.ocr_preprocess_orientation,
        adaptive_threshold=settings.ocr_preprocess_threshold,
    )
    return EasyOCRService(
        languages=settings.ocr_language_list,
        gpu=settings.ocr_gpu,
        preprocessor=preprocessor,
        parser=SpineTextParser(),
    )

@lru_cache(maxsize=1)
def get_confidence_router(
    settings: Settings = Depends(get_settings),
) -> BaseConfidenceRouter:
    """Provides the confidence router with the configured threshold."""
    return ThresholdConfidenceRouter(
        threshold=settings.ocr_confidence_threshold,
    )


@lru_cache(maxsize=1)
def get_review_queue_service(
    settings: Settings = Depends(get_settings),
) -> BaseReviewQueueService:
    """Provides the Redis-backed review queue for HITL corrections."""
    return RedisReviewQueueService(
        redis_url=settings.redis_url,
        ttl_seconds=settings.review_session_ttl_seconds,
    )

## NOTE: What @lru_cache(maxsize=1) does:
# - Caches the result of a function call
# - On the first call to that function, it executes and stores the result
# - On subsequent calls, it returns the cached result (no re-execution)

# In your app, the flow is:
# 1. App starts → FastAPI loads routes but does NOT call dependencies
# 2. First request arrives that needs review_queue dependency
# 3. FastAPI calls get_review_queue_service() → it executes, creates RedisReviewQueueService, and caches it
# 4. Second request needs review_queue → FastAPI calls get_review_queue_service() again, but @lru_cache returns the cached instance (no new object created)

# So it's "lazy loading" — the service is created on first use, not startup.

# Summary:
# 1. @lru_cache = "create once, reuse forever" (per process)
# 2. Not automatic on startup — triggered by first route that uses it
# 3. Optionally, you can call these functions manually in the lifespan to pre-warm them

@lru_cache(maxsize=1)
def get_result_merger() -> BaseResultMerger:
    """Provides the result merger (no configuration needed)."""
    return DefaultResultMerger()


@lru_cache(maxsize=1)
def get_vector_store_service(
    settings: Settings = Depends(get_settings),
) -> BaseVectorStoreService:
    """Provides the Qdrant vector store client."""
    from app.services.embeddings import SentenceTransformerEmbeddingService

    return QdrantVectorStoreService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            similarity_threshold=settings.qdrant_similarity_threshold,
            embedding_service=SentenceTransformerEmbeddingService(
                model_name=settings.embedding_model_name
                ),
    )


@lru_cache(maxsize=1)
def get_catalog_service(
    settings: Settings = Depends(get_settings),
) -> BaseCatalogService:
    """Provides the LLM-powered catalog/genre-coding service."""
    return LLMCatalogService(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
        batch_size=settings.llm_catalog_batch_size,
    )


@lru_cache(maxsize=1)
def get_recommendation_service(
    settings: Settings = Depends(get_settings),
) -> BaseRecommendationService:
    """Provides the LLM-powered recommendation service."""
    return LLMRecommendationService(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        max_tokens=settings.anthropic_max_tokens,
    )
