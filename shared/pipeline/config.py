"""
Pipeline configuration - API keys, retrieval parameters, LLM settings.
"""

import os
import logging

logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration with validation"""

    # API Keys
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    COLLECTION = os.getenv("QDRANT_COLLECTION", "ev_kb")

    # Retrieval parameters
    TOP_K = int(os.getenv("TOP_K", "12"))
    TOP_K_EXPAND = int(os.getenv("TOP_K_EXPAND", "50"))

    # Hybrid scoring weights
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.65"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.35"))

    # Context building
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
    CHARS_PER_TOKEN_ESTIMATE = float(os.getenv("CHARS_PER_TOKEN_ESTIMATE", "4.0"))

    # LLM parameters (OpenAI hosted)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1-chat-latest")
    # Use MAX_COMPLETION_TOKENS for models that require explicit completion token limit
    MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", os.getenv("MAX_TOKENS", "400")))
    # Backwards-compatible alias for older code
    MAX_TOKENS = MAX_COMPLETION_TOKENS
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))

    # Query processing
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
    MAX_REWRITE_VARIANTS = int(os.getenv("MAX_REWRITE_VARIANTS", "3"))

    # Timeouts
    QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))

    # Cache settings
    ENABLE_EMBEDDING_CACHE = os.getenv("ENABLE_EMBEDDING_CACHE", "true").lower() == "true"
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

    # Parallel retrieval settings
    ENABLE_PARALLEL_RETRIEVAL = os.getenv("ENABLE_PARALLEL_RETRIEVAL", "true").lower() == "true"
    MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "5"))

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration"""
        required = {
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
            "QDRANT_API_KEY": cls.QDRANT_API_KEY,
            "QDRANT_URL": cls.QDRANT_URL,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Validate weight sum
        if not 0.99 <= (cls.VECTOR_WEIGHT + cls.BM25_WEIGHT) <= 1.01:
            logger.warning(
                f"Scoring weights don't sum to 1.0: "
                f"VECTOR={cls.VECTOR_WEIGHT}, BM25={cls.BM25_WEIGHT}"
            )
