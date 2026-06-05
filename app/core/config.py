# ─────────────────────────────────────────────────────────────────────────────
# app/core/config.py
#
# Central configuration module.
# All environment variables are read ONCE here via Pydantic's BaseSettings.
# Every other module imports `get_settings()` – never os.environ directly.
#
# SOLID (Dependency Inversion):
#   Services depend on this Settings object (an abstraction), not on raw env
#   vars scattered throughout the codebase. Swapping config sources (e.g. from
#   .env file to AWS Secrets Manager) requires changes only in this file.
# ─────────────────────────────────────────────────────────────────────────────

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application-wide settings.

    Pydantic-settings automatically reads each field from:
      1. A matching environment variable (case-insensitive), OR
      2. A `.env` file in the project root, OR
      3. The default value defined here.

    Type annotations enforce validation – e.g. assigning a string to an int
    field raises a clear error at startup rather than silently misbehaving.
    """


    # ── Application ──────────────────────────────────────────────────────────
    env: str = "development"  ## This can be "development", "production", or "test" – used to toggle debug features
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    
    load_models_on_startup: bool = False   # NOTE: Since we're using celery workers for all backend tasks.
                                           #       Warn loading and caching models on app startup would just
                                           #       consume unnecessary memory and compute. Which we won't use
                                           #       at all. 

    # ── Anthropic LLM ─────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4"
    anthropic_max_tokens: int = 2048
    # Books per LLM batch call in the catalog pipeline (keeps token usage bounded)
    llm_catalog_batch_size: int = 25
    
    # ── Gemma-2B LLM : light weight LLM with on system inference ──────────────
    gemma_model_path: str = "/app/models/llama/gemma-2b-it.Q4_K_M.gguf"   
                                                               # Path to Gemma model in directory
    gemma_n_ctx: int = 1024                                    # Model Context Window
    gemma_n_thread: int = 4                                    # Number of threads to be used for inference
    gemma_n_batch: int = 256                                   # Number of input prompt tokens to be processed internally by model
    gemma_n_gpu_layers: int = 0                                # Number of model layers to be offloaded to GPU
    
    # Configs for Text Summarization Task:
    gemma_max_tokens: int = 160                                # Max tokens that model generates on inference 
    gemma_llm_temp: float = 0.9                                # Temperture param for Gemma LLM

    embedding_model_path:str = "/app/models/all-MiniLM-L6-v2"  # Using - all-MiniLM-L6-v2
    genre_saved_embediings_file_path: str = "/app/models/saved_genre_embeddings/genre_embeddings.npy"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name_reader: str = "books"
    qdrant_collection_name_librarian: str = "library_books"
    # Cosine similarity score (0-1) above which a Qdrant hit is a cache hit
    qdrant_similarity_threshold: float = 0.92
    # Sentence-transformers model for book title+author embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    # Redis TTL for HITL review sessions in seconds (default = 24 h)
    review_session_ttl_seconds: int = 86_400

    # ── OCR ───────────────────────────────────────────────────────────────────
    # Books whose OCR confidence falls below this threshold are sent to the
    # human-in-the-loop review queue instead of being auto-accepted.
    ocr_confidence_threshold: float = 0.75
    ocr_languages: str = "en"           # Comma-separated; parsed as list below
    
    ocr_device: str = "cpu"
    
    ## PARAMETERS of PaddleOCR -> TextDetection : model_name = "PP-OCRv5_mobile_det" 
    detector_threshold: float = 0.1
    detector_box_threshold: float = 0.3
    detector_unclip_ratio: float = 1.8
    
    # After text region detection using PaddleOCR TextDetection. We need to filter out text boxes which
    # are very close to image borders. "min_box_distance_px" tells the minimum distance the text region
    # must maintain
    min_box_distance_px: int = 5     
    
    # Parameter for tunning filtearation of recognised texts from text recognition results
    # keeping text which is recognised with good confidence score
    min_conf_thresh_text_rec: float = 0.7      
    
    # Google Books API for accessing book data based on OCR extracted text
    google_books_api_key: str = ""
    
    # Use GPU for EasyOCR inference (requires CUDA-enabled machine)
    ocr_gpu: bool = False
    # Preprocessing toggles — disable individual steps for debugging
    ocr_preprocess_upscale: bool = True
    ocr_preprocess_orientation: bool = False    # PaddleOCR is great with unprocessed simple RGB images
    ocr_preprocess_threshold: bool = False      # & No need for normalising orientationa
 
    
    # ── Object detection ──────────────────────────────────────────────────────
    # Empty string means "use torchvision pretrained weights".
    detection_model_weights: str = ""
    num_detection_classes: int = 0         # On sending fine-tuned model, we can set the number of 
                                           # classes that the model detects using this parameter
                                           # For pre-downloaded Faster-RCNN Model it remains 'None'
    
    # Minimum Faster-RCNN score to keep a detection (0.0–1.0)
    detection_threshold: float = 0.70
    # IoU threshold for Non-Maximum Suppression
    detection_nms_iou_threshold: float = 0.40
    # width to height difference of bounding box around detected book must be greater than "200" for it be a legit book crop
    width_to_height_diff_of_book: int = 200
    # Padding to be applied around detected bounding box before crop from original image
    padding_around_book_crop: int = 2
    # PyTorch device: "cpu" or "cuda"
    detection_device: str = "cpu"

    # ── File storage ──────────────────────────────────────────────────────────
    upload_dir: str = "/app/uploads"
    max_upload_size_bytes: int = 20 * 1024 * 1024   # 20 MB default

    # ── Pydantic-settings configuration ───────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",    # Load from .env in current working directory (project root)
        env_file_encoding="utf-8",
        case_sensitive=False,       # ENV_VAR and env_var both work
        extra="ignore",             # Silently ignore unknown env vars
        protected_namespaces=(),    # Disable protected namespace warnings (some deps use model_kwargs)
    )
    
    ## NOTE: SettingsConfigDict() DOES NOT parse the .env file itself; it just tells Pydantic where 
    ## to look for values of fields like 'detection_threshold', 'detection_device' etc, 
    ## when instantiating Settings. The actual parsing happens when get_settings() calls Settings().
    ## The Pydantic parses the .env file into memory and uses it to assign values of fields of the Settings object. 
    ## The .env file is NOT parsed on every call to get_settings(), but only once when the Settings object is first created (and then cached by @lru_cache).
    ## It DOES NOT store the parsed values in os.environ; they are only accessible via the Settings object returned by get_settings().

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def ocr_language_list(self) -> list[str]:
        """Parse the comma-separated OCR languages string into a Python list."""
        return [lang.strip() for lang in self.ocr_languages.split(",") if lang.strip()]

    @property
    def is_production(self) -> bool:
        """Convenience flag – disables debug features in production."""
        return self.env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    Using @lru_cache means the .env file is parsed only once per process
    lifetime, not on every call. In tests, call `get_settings.cache_clear()`
    before patching env vars to force re-initialisation.
    """
    return Settings()




# How does .env Files loading work?
# Flow:
# 1. When get_settings() is called, Pydantic instantiates Settings
# 2. SettingsConfigDict tells Pydantic: "Read variables from .env file"
# 3. Pydantic parses the .env file (key=value format) into memory (NOT os.environ)
# 4. Each field (like `anthropic_api_key`) in the Settings class is validated and assigned from the parsed values
# 5. Values are ONLY accessible via the Settings object, not os.environ

# ❌ This will NOT work (values not in os.environ):
# import os
# print(os.environ.get('ANTHROPIC_API_KEY'))  # None

# ✅ This IS the correct way:
# settings = get_settings()
# print(settings.anthropic_api_key)  # "sk-ant-xxx" ✅

# CACHING: @lru_cache ensures Settings is instantiated only once per process.
# In tests, call get_settings.cache_clear() before changing env vars.