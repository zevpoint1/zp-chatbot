"""
Centralized configuration for the chatbot.
All magic numbers and tunable parameters are defined here.
Can be overridden via environment variables.
"""

import os


class AppConfig:
    """Application-wide configuration constants."""

    # --------------------------------------------------
    # Conversation & Memory Settings
    # --------------------------------------------------
    # Maximum messages to store in conversation history (increased from 10 to 16 for longer sales conversations)
    MAX_STORED_MESSAGES = int(os.getenv("MAX_STORED_MESSAGES", "16"))

    # Number of recent messages to include in LLM context (should be even for user/assistant pairs)
    # Increased from 6 to 10 to maintain context through typical sales flow
    CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "10"))

    # Default top_k for RAG retrieval in HTTP handler
    DEFAULT_RAG_TOP_K = int(os.getenv("DEFAULT_RAG_TOP_K", "8"))

    # --------------------------------------------------
    # Rate Limiting
    # --------------------------------------------------
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "500"))

    # Seconds to wait before retrying after rate limit
    RATE_LIMIT_RETRY_AFTER = int(os.getenv("RATE_LIMIT_RETRY_AFTER", "60"))

    # Hours after which inactive rate limit entries are cleaned up
    RATE_LIMIT_CLEANUP_HOURS = int(os.getenv("RATE_LIMIT_CLEANUP_HOURS", "2"))

    # --------------------------------------------------
    # Reranking
    # --------------------------------------------------
    # Maximum candidates to send to reranker (performance optimization)
    MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "20"))

    # --------------------------------------------------
    # LLM Retry Settings
    # --------------------------------------------------
    # Number of retries for LLM API calls
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # --------------------------------------------------
    # Display/Preview Limits
    # --------------------------------------------------
    # Characters to show in context preview for fallback responses
    CONTEXT_PREVIEW_LENGTH = int(os.getenv("CONTEXT_PREVIEW_LENGTH", "500"))

    # Characters to show in chunk preview for debugging
    CHUNK_PREVIEW_LENGTH = int(os.getenv("CHUNK_PREVIEW_LENGTH", "200"))

    # Characters to show in nudge log preview
    NUDGE_LOG_PREVIEW_LENGTH = int(os.getenv("NUDGE_LOG_PREVIEW_LENGTH", "50"))

    # --------------------------------------------------
    # Query Processing
    # --------------------------------------------------
    # Maximum words for a query to be considered "vague" (eligible for enrichment)
    VAGUE_QUERY_MAX_WORDS = int(os.getenv("VAGUE_QUERY_MAX_WORDS", "6"))

    # --------------------------------------------------
    # HTTP Connection Pool
    # --------------------------------------------------
    HTTP_POOL_CONNECTIONS = int(os.getenv("HTTP_POOL_CONNECTIONS", "10"))
    HTTP_POOL_MAXSIZE = int(os.getenv("HTTP_POOL_MAXSIZE", "20"))
    HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))

    # --------------------------------------------------
    # Azure Table Storage
    # --------------------------------------------------
    CHAT_HISTORY_TABLE = os.getenv("CHAT_HISTORY_TABLE", "ChatHistory")
