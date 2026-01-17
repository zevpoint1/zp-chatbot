"""
query_pipeline.py - Backward compatibility wrapper.

This file maintains backward compatibility by re-exporting all public APIs
from the modular pipeline package. The actual implementation is now split into:

- shared/pipeline/config.py      - Pipeline configuration (Config class)
- shared/pipeline/models.py      - Data classes (QueryMetrics, PipelineResult)
- shared/pipeline/preprocessing.py - Query validation, normalization, rewriting
- shared/pipeline/retrieval.py   - Qdrant search, BM25, hybrid scoring
- shared/pipeline/generation.py  - LLM calls, context building, response parsing
- shared/pipeline/orchestrator.py - Main answer_question function

All existing imports will continue to work:
    from shared.query_pipeline import answer_question, PipelineResult, Config
"""

# Re-export prompt_manager functions that were previously accessed via query_pipeline
from shared.prompt_manager import detect_intent, build_prompt

# Re-export everything from pipeline modules for backward compatibility
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
    # Internal but re-exported for compatibility
    _parse_llm_rewrite_response,
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
    _extract_hit_id,
)
from shared.pipeline.generation import (
    get_tokenizer,
    count_tokens,
    estimate_tokens,
    build_context_and_sources,
    call_openai_chat_api,
    parse_answer_nudge,
    generate_llm_failure_fallback,
    generate_fallback_response,
    TIKTOKEN_AVAILABLE,
)
from shared.pipeline.orchestrator import answer_question, main

# Backward compatibility aliases
_generate_llm_failure_fallback = generate_llm_failure_fallback
_generate_fallback_response = generate_fallback_response
_parse_answer_nudge = parse_answer_nudge

__all__ = [
    # Prompt manager
    "detect_intent",
    "build_prompt",
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
    "_parse_llm_rewrite_response",
    # Retrieval
    "get_http_session",
    "embed_cached",
    "qdrant_search",
    "extract_search_results",
    "build_qdrant_filter",
    "retrieve",
    "rerank_hits",
    "RERANK_ENABLED",
    "_extract_hit_id",
    # Generation
    "get_tokenizer",
    "count_tokens",
    "estimate_tokens",
    "build_context_and_sources",
    "call_openai_chat_api",
    "parse_answer_nudge",
    "generate_llm_failure_fallback",
    "generate_fallback_response",
    "TIKTOKEN_AVAILABLE",
    # Backward compat aliases
    "_generate_llm_failure_fallback",
    "_generate_fallback_response",
    "_parse_answer_nudge",
    # Orchestrator
    "answer_question",
    "main",
]

# CLI entry point
if __name__ == "__main__":
    main()
