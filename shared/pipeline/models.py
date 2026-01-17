"""
Data models for the RAG pipeline.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class QueryMetrics:
    """Metrics collected during pipeline execution."""
    retrieved_count: int = 0
    reranked_count: int = 0
    chunks_used: int = 0
    search_time_ms: float = 0.0
    preprocessing_time: float = 0.0
    llm_rewrite_time: float = 0.0
    retrieval_time: float = 0.0
    rerank_time: float = 0.0
    context_tokens: int = 0
    llm_generation_time: float = 0.0
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Result returned by the RAG pipeline."""
    answer: str
    sources: List[str]
    metrics: QueryMetrics
    filters: Dict[str, str]
    rewritten_queries: List[str]
    retrieved_chunks: List[str]
    intents: List[str] = None
    confidence_score: Optional[float] = None
    search_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    nudge: Optional[str] = None  # Optional follow-up message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "nudge": self.nudge,
            "sources": self.sources,
            "metrics": self.metrics.to_dict(),
            "filters": self.filters,
            "rewritten_queries": self.rewritten_queries,
            "confidence_score": self.confidence_score,
        }


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass
