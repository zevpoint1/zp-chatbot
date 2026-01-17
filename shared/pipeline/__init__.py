"""
Pipeline module - modular RAG pipeline components.

This module splits the monolithic query_pipeline.py into smaller, focused modules:
- config.py: Pipeline-specific configuration (Config class)
- models.py: Data classes (QueryMetrics, PipelineResult, ValidationError)
- preprocessing.py: Query validation, normalization, rewriting
- retrieval.py: Qdrant search, BM25, hybrid scoring
- generation.py: LLM calls, response parsing, fallbacks
- orchestrator.py: Main answer_question function

For backward compatibility, all public APIs are re-exported from this __init__.py
"""

# Re-export everything for backward compatibility
from shared.pipeline.config import Config
from shared.pipeline.models import QueryMetrics, PipelineResult, ValidationError
from shared.pipeline.preprocessing import (
    validate_question,
    normalize_query,
    extract_filters,
    rewrite_query_simple,
    rewrite_query_with_llm,
    extract_context_from_history,
    enrich_query_with_context,
    SYNONYM_MAP,
    PRODUCT_PATTERNS,
    VEHICLE_PATTERNS,
)
from shared.pipeline.retrieval import (
    get_http_session,
    embed_cached,
    qdrant_search,
    extract_search_results,
    build_qdrant_filter,
    retrieve,
    rerank_hits,
    RERANK_ENABLED,
)
from shared.pipeline.generation import (
    get_tokenizer,
    count_tokens,
    estimate_tokens,
    build_context_and_sources,
    call_openai_chat_api,
)
from shared.pipeline.orchestrator import answer_question, main

__all__ = [
    # Config
    "Config",
    # Models
    "QueryMetrics",
    "PipelineResult",
    "ValidationError",
    # Preprocessing
    "validate_question",
    "normalize_query",
    "extract_filters",
    "rewrite_query_simple",
    "rewrite_query_with_llm",
    "extract_context_from_history",
    "enrich_query_with_context",
    "SYNONYM_MAP",
    "PRODUCT_PATTERNS",
    "VEHICLE_PATTERNS",
    # Retrieval
    "get_http_session",
    "embed_cached",
    "qdrant_search",
    "extract_search_results",
    "build_qdrant_filter",
    "retrieve",
    "rerank_hits",
    "RERANK_ENABLED",
    # Generation
    "get_tokenizer",
    "count_tokens",
    "estimate_tokens",
    "build_context_and_sources",
    "call_openai_chat_api",
    # Orchestrator
    "answer_question",
    "main",
]
