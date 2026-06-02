from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def get_worker_services_Det_OCR_Mer_Rou() -> Dict:
    """
    Instantiate and cache Detection, OCR, Merger and Router services, to be 
    used by Celery worker processes.

    This function is intentionally lru_cache-decorated so that each worker
    process loads models once at first call (or when explicitly primed on
    worker startup) and reuses the same instances for subsequent tasks.

    Returns:
        Dict mapping service name -> service instance.
    """
    # Import inside the function to avoid loading heavy model modules at
    # import time in processes that don't need them.
    from app.core.config import get_settings
    from app.services.detection import FasterRCNNDetectionService
    from app.services.ocr import EasyOCRService
    from app.services.ocr_preprocessing import OCRSpinePreprocessor
    from app.services.text_parser import SpineTextValidator
    from app.services.confidence_router import ThresholdConfidenceRouter

    settings = get_settings()

    return {
        "detector": FasterRCNNDetectionService(
            model_weights_path=settings.detection_model_weights or None,
            num_detection_classes=settings.num_detection_classes,
            upload_dir=settings.upload_dir,
            detection_threshold=settings.detection_threshold,
            nms_iou_threshold=settings.detection_nms_iou_threshold,
            width_to_height_diff_of_book = settings.width_to_height_diff_of_book,
            padding_around_book_crop = settings.padding_around_book_crop,
            device=settings.detection_device,
        ),
        "ocr": EasyOCRService(
            languages=settings.ocr_language_list,
            gpu=settings.ocr_gpu,
            preprocessor=OCRSpinePreprocessor(
                upscale=settings.ocr_preprocess_upscale,
                fix_orientation=settings.ocr_preprocess_orientation,
                adaptive_threshold=settings.ocr_preprocess_threshold,
            ),
            book_validator=SpineTextValidator(
                api_key=settings.google_books_api_key,
            ),
            device=settings.ocr_device,
            detector_threshold=settings.detector_threshold,
            detector_box_threshold=settings.detector_box_threshold,
            detector_unclip_ratio=settings.detector_unclip_ratio,
            min_box_distance_px = settings.min_box_distance_px,
            min_conf_thresh_text_rec = settings.min_conf_thresh_text_rec,
        ),
        "router": ThresholdConfidenceRouter(
            threshold=settings.ocr_confidence_threshold,
        ),
    }



@lru_cache(maxsize=1)
def get_worker_services_REC() -> Dict:
    """
    Instantiate and cache LLM Recommendation service, to be used by Celery 
    worker processes.

    This function is intentionally lru_cache-decorated so that each worker
    process loads models once at first call (or when explicitly primed on
    worker startup) and reuses the same instances for subsequent tasks.

    Returns:
        Dict mapping service name -> service instance.
    """
    # Import inside the function to avoid loading heavy model modules at
    # import time in processes that don't need them.
    from app.core.config import get_settings
    from app.services.recommendation import GemmaRecommendationService
    from app.services.llm_client import GemmaLLMClient

    settings = get_settings()

    return {
        "recommender": GemmaRecommendationService(
            llm_client= GemmaLLMClient(
                model_path=settings.gemma_model_path ,
                context_window_len=(settings.gemma_n_ctx) ,
                n_threads=(settings.gemma_n_thread) ,
                batch_size=(settings.gemma_n_batch) ,
                offload_n_gpu_layers=(settings.gemma_n_gpu_layers) ,
                
                max_inf_tokens=(settings.gemma_max_tokens) ,
                llm_temperature=(settings.gemma_llm_temp) ,
                ),
            embedding_model_path=settings.embedding_model_path,
            genre_emb_file_path=settings.genre_saved_embediings_file_path,
        ),
    }

@lru_cache(maxsize=1)
def get_worker_services_Reader_VS() -> Dict:
    """
    Instantiate and cache Vector Store service, to be used by Celery worker processes.

    This function is intentionally lru_cache-decorated so that each worker
    process loads models once at first call (or when explicitly primed on
    worker startup) and reuses the same instances for subsequent tasks.

    Returns:
        Dict mapping service name -> service instance.
    """
    # Import inside the function to avoid loading heavy model modules at
    # import time in processes that don't need them.
    from app.core.config import get_settings
    from app.services.vector_store import QdrantVectorStoreService
    from app.services.embeddings import SentenceTransformerEmbeddingService

    settings = get_settings()
    
    # We are using same embedding model for recommendation work. 
    # Let's use that same model instance  here...
    rec_serv = get_worker_services_REC()
    embedding_model = rec_serv["recommender"]._emb_model 
    
    return {
        "vector_store": QdrantVectorStoreService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name_reader,
            similarity_threshold=settings.qdrant_similarity_threshold,
            embedding_service=SentenceTransformerEmbeddingService(
                model=embedding_model,
                ),
        ),
    }



@lru_cache(maxsize=1)
def get_worker_services_Librarian_VS() -> Dict:
    """
    Instantiate and cache Vector Store service, to be used by Celery worker processes.

    This function is intentionally lru_cache-decorated so that each worker
    process loads models once at first call (or when explicitly primed on
    worker startup) and reuses the same instances for subsequent tasks.

    Returns:
        Dict mapping service name -> service instance.
    """
    # Import inside the function to avoid loading heavy model modules at
    # import time in processes that don't need them.
    from app.core.config import get_settings
    from app.services.vector_store import QdrantVectorStoreService
    from app.services.embeddings import SentenceTransformerEmbeddingService

    settings = get_settings()
    
    # We are using same embedding model for recommendation work. 
    # Let's use that same model instance  here...
    rec_serv = get_worker_services_REC()
    embedding_model = rec_serv["recommender"]._emb_model 
    
    return {
        "vector_store": QdrantVectorStoreService(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name_librarian,
            similarity_threshold=settings.qdrant_similarity_threshold,
            embedding_service=SentenceTransformerEmbeddingService(
                model=embedding_model,
                ),
        ),
    }
    
    
    

@lru_cache(maxsize=1)
def get_worker_services_CAT_ESS() -> Dict:         # Services necessary for Catalog Generation
    """
    Instantiate and Plain Catalog Generation Service, to be used by Celery worker 
    processes.

    This function is intentionally lru_cache-decorated so that each worker
    process loads models once at first call (or when explicitly primed on
    worker startup) and reuses the same instances for subsequent tasks.

    Returns:
        Dict mapping service name -> service instance.
    """
    # Import inside the function to avoid loading heavy model modules at
    # import time in processes that don't need them.
    from app.core.config import get_settings
    from app.services.review_queue import RedisReviewQueueService
    from app.services.catalog import PlainCatalogService
    from app.services.job_store import JobStore
    
    settings = get_settings()
    # We are using same embedding model for recommendation work. 
    # Let's use that same model instance  here...
    rec_serv = get_worker_services_REC()
    embedding_model = rec_serv["recommender"]._emb_model 
    
    
    return {
        "review_queue": RedisReviewQueueService(
            redis_url=settings.redis_url,
            ttl_seconds=settings.review_session_ttl_seconds,
        ),
        "catalog": PlainCatalogService(
                emb_model=embedding_model,
                genre_emb_file_path=settings.genre_saved_embediings_file_path,
            ),
        "job_store": JobStore(redis_url=settings.redis_url),
        }
    
    