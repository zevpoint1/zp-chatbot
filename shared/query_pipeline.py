"""
query_pipeline.py
Enhanced RAG pipeline with:
- Robust input validation and error handling
- Configurable scoring weights and parameters
- Embedding cache for performance
- Better observability (metrics, timing)
- Structured outputs from LLM
- Improved query rewriting with fallbacks
- Token-aware context building
- Type safety improvements
"""

import sys
import os
import re
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.prompt_manager import detect_intent, build_prompt
from shared.embed_text import embed
from shared.hybrid_search import run_keyword_search

import requests
from openai import OpenAI, OpenAIError

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available, falling back to character-based token estimation")

# Local imports -- ensure these modules exist in your project
from shared.embed_text import embed
from shared.hybrid_search import run_keyword_search

# Optional reranker
try:
    from shared.reranker import rerank
    RERANK_ENABLED = True  # ← FIXED: Was False
except Exception as e:
    logging.warning(f"Reranker not available: {e}")
    RERANK_ENABLED = False

# ========================
# Logging Configuration
# ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================
# Connection Pool for HTTP Requests
# ========================
# Create a session with connection pooling for better performance
_http_session = None

def get_http_session():
    """Get or create HTTP session with connection pooling"""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
            pool_block=False
        )
        _http_session.mount('http://', adapter)
        _http_session.mount('https://', adapter)
        logger.info("HTTP session with connection pooling initialized")
    return _http_session


# ========================
# Configuration & Constants
# ========================
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


# ========================
# Data Classes
# ========================
@dataclass
class QueryMetrics:
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

# ========================
# Input Validation
# ========================
class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_question(question: str) -> str:
    """Validate and sanitize user question"""
    if not question:
        raise ValidationError("Question cannot be empty")
    
    question = question.strip()
    
    if not question:
        raise ValidationError("Question cannot be only whitespace")
    
    if len(question) > Config.MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Question too long ({len(question)} chars). "
            f"Maximum allowed: {Config.MAX_QUERY_LENGTH}"
        )
    
    # Check for potential injection attacks and reject
    suspicious_patterns = [
        r"<script[^>]*>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
        r"eval\s*\(",
        r"expression\s*\(",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            logger.warning(f"Suspicious pattern detected and rejected: {pattern}")
            raise ValidationError(
                "Invalid input detected. Please rephrase your question without special characters or code."
            )

    return question


# ========================
# Query Preprocessing
# ========================
SYNONYM_MAP = {
    "vs": "versus",
    "vs.": "versus",
    "svc": "service",
    "svc.": "service",
    "kpi": "key performance indicator",
    "kpis": "key performance indicators",
    "cpu": "central processing unit",
    "gpu": "graphics processing unit",
    "ram": "random access memory",
    "api": "application programming interface",
    "ui": "user interface",
    "ux": "user experience",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "ev": "electric vehicle",
    "evs": "electric vehicles",
    # Add domain-specific expansions here
}

# Product and vehicle patterns for context extraction
PRODUCT_PATTERNS = [
    r'\b(aveo\s*(pro|x1|3\.6|plus)?)\b',
    r'\b(dash\s*(aio)?)\b',
    r'\b(spyder)\b',
    r'\b(duos\s*(7\.5|22)?)\b',
    r'\b(polar\s*(pro|x1|max)?)\b',
    r'\b(nova\s*(60|120|240)?)\b',
    r'\b(titan)\b',
    r'\b(navigator)\b',
]

VEHICLE_PATTERNS = [
    r'\b(nexon|curvv|tiago|tigor|punch)\b',
    r'\b(xuv\s*400|e20|everito|be6|xev\s*9e)\b',
    r'\b(zs\s*ev|comet|windsor|cyberster|m9)\b',
    r'\b(kona|ioniq|creta)\b',
    r'\b(ec3|atto|e6|seal|emax)\b',
    r'\b(ev6|carens)\b',
]


def extract_context_from_history(
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, str]:
    """
    Extract relevant entities from conversation history for query enrichment.
    Returns dict with 'product' and 'vehicle' keys if found.
    """
    context = {'product': None, 'vehicle': None}

    if not conversation_history:
        return context

    # Combine all messages for context extraction (most recent first)
    all_text = ""
    for msg in reversed(conversation_history[-6:]):  # Last 3 exchanges
        all_text += " " + msg.get('content', '').lower()

    # Extract most recent product mention
    for pattern in PRODUCT_PATTERNS:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            context['product'] = match.group(0).strip()
            break

    # Extract most recent vehicle mention
    for pattern in VEHICLE_PATTERNS:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            context['vehicle'] = match.group(0).strip()
            break

    return context


def enrich_query_with_context(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Enrich a short/vague query with context from conversation history.
    This improves retrieval for follow-up questions like "what's the price".
    """
    if not conversation_history:
        return query

    query_lower = query.lower().strip()

    # Check if query is vague (short or lacks specific entities)
    query_words = query_lower.split()
    is_vague = (
        len(query_words) <= 6 and
        not any(re.search(p, query_lower) for p in PRODUCT_PATTERNS) and
        not any(re.search(p, query_lower) for p in VEHICLE_PATTERNS)
    )

    if not is_vague:
        return query

    # Extract context from history
    context = extract_context_from_history(conversation_history)

    # Enrich query with context
    enrichments = []
    if context['product']:
        enrichments.append(context['product'])
    if context['vehicle']:
        enrichments.append(context['vehicle'])

    # Also detect what type of info user is asking for and add relevant terms
    # This helps when user asks vague follow-ups like "pricing" or "details"
    info_type_expansions = {
        # Price-related
        ("price", "pricing", "cost", "rate", "kitna", "kitne", "quote"): "price cost rs",
        # Installation-related
        ("install", "installation", "setup", "fitting"): "installation service requirements",
        # Warranty-related
        ("warranty", "guarantee"): "warranty years coverage",
        # Specification-related
        ("spec", "specification", "details", "features", "info"): "specifications features",
        # Delivery-related
        ("delivery", "shipping", "dispatch"): "delivery dispatch days",
    }

    for keywords, expansion in info_type_expansions.items():
        if any(kw in query_lower for kw in keywords):
            enrichments.append(expansion)
            break

    if enrichments:
        enriched = f"{query} {' '.join(enrichments)}"
        logger.info(f"Query enriched with context: '{query}' -> '{enriched}'")
        return enriched

    return query


# Compiled patterns for efficiency
DATE_PATTERN = re.compile(
    r"date\s*[:=]\s*([0-9]{4}(?:-[0-9]{2}(?:-[0-9]{2})?)?)",
    re.IGNORECASE
)
SOURCE_PATTERN = re.compile(
    r"source\s*[:=]\s*([A-Za-z0-9_\-\.]+)",
    re.IGNORECASE
)
TYPE_PATTERN = re.compile(
    r"type\s*[:=]\s*([A-Za-z0-9_\-]+)",
    re.IGNORECASE
)


def normalize_query(q: str) -> str:
    """
    Deterministic normalization of query text.
    - Trim whitespace
    - Normalize whitespace to single spaces
    - Remove disruptive punctuation
    - Lowercase
    - Expand synonyms
    """
    if not q:
        return q
    
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    
    # Remove punctuation that fragments tokens
    # Keep: ?, !, :, /, #, -, . (for filters, dates, decimals)
    q = re.sub(r"[^\w\s\?\!:/#\-\.\@]", " ", q)
    q = q.lower()
    
    # Expand acronyms and synonyms with word boundary matching
    for acronym, expansion in SYNONYM_MAP.items():
        pattern = rf"\b{re.escape(acronym)}\b"
        q = re.sub(pattern, expansion, q)
    
    q = re.sub(r"\s+", " ", q).strip()
    return q


def extract_filters(q: str) -> Tuple[str, Dict[str, str]]:
    """
    Extract structured filters from query.
    Supports: date:YYYY-MM-DD, source:filename, type:doctype
    Returns: (cleaned_query, filters_dict)
    """
    filters: Dict[str, str] = {}
    
    # Extract date filter
    date_match = DATE_PATTERN.search(q)
    if date_match:
        filters["date"] = date_match.group(1)
        q = DATE_PATTERN.sub("", q)
    
    # Extract source filter
    source_match = SOURCE_PATTERN.search(q)
    if source_match:
        filters["source"] = source_match.group(1)
        q = SOURCE_PATTERN.sub("", q)
    
    # Extract type filter
    type_match = TYPE_PATTERN.search(q)
    if type_match:
        filters["type"] = type_match.group(1)
        q = TYPE_PATTERN.sub("", q)
    
    # Clean up remaining whitespace
    q = re.sub(r"\s+", " ", q).strip()
    
    if filters:
        logger.info(f"Extracted filters: {filters}")
    
    return q, filters


def rewrite_query_simple(q: str) -> str:
    """
    Deterministic query rewrite for retrieval optimization.
    - Remove common stopwords
    - Preserve key semantic tokens
    - Add query hints for question-like inputs
    - Expand installation queries to include service document terms
    """
    q = normalize_query(q)

    # Minimal stopword list - keep words that might be semantically important
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were",
        "of", "in", "on", "at", "to", "for", "with",
    }

    tokens = [t for t in q.split() if t not in stopwords or len(t) > 3]
    rewritten = " ".join(tokens)

    # Add retrieval hint for question-style queries
    if "?" in q or (tokens and tokens[0] in {"how", "what", "why", "when", "who", "where"}):
        rewritten = f"{rewritten} query"

    # Expand installation queries to include service-related terms
    if "install" in q.lower():
        rewritten = f"{rewritten} installation service cost booking rzp"

    # Expand wiring/electrical uncertainty queries to retrieve installation services
    wiring_keywords = ["wiring", "electrical", "wire", "cable", "mcb", "earthing", "phase", "socket", "power supply"]
    if any(kw in q.lower() for kw in wiring_keywords):
        rewritten = f"{rewritten} installation service survey site assessment requirements"

    # Expand pricing/cost queries to match "Price:" in documents
    pricing_keywords = ["pricing", "price", "cost", "rate", "quote", "how much", "kitna", "kitne"]
    if any(kw in q.lower() for kw in pricing_keywords):
        rewritten = f"{rewritten} price cost rs rupees"

    return rewritten.strip()


# ========================
# LLM-based Query Rewriting
# ========================
def rewrite_query_with_llm(
    user_question: str,
    max_variants: int = Config.MAX_REWRITE_VARIANTS
) -> List[str]:
    """
    Use OpenAI hosted chat API to generate optimized retrieval queries.
    Returns list of rewritten queries with robust parsing and fallbacks.
    """
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Clear, structured prompt for JSON output
    system_prompt = (
        "You are a query rewriting assistant. Generate alternative search queries "
        "optimized for retrieval. Output ONLY valid JSON - no explanation, no markdown."
    )
    
    user_message = (
        f"Rewrite this question into {max_variants} short retrieval queries.\n\n"
        f"Question: {user_question}\n\n"
        f"Output format: {{\"queries\": [\"query1\", \"query2\", \"query3\"]}}\n"
        f"Rules:\n"
        f"- Each query should be 3-10 words\n"
        f"- Focus on key concepts and entities\n"
        f"- Remove question words (what, how, etc.)\n"
        f"- Output ONLY the JSON object, nothing else"
    )
    
    try:
        start_time = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=300,
            timeout=Config.OPENAI_TIMEOUT
        )
        
        elapsed = time.time() - start_time
        logger.debug(f"LLM rewrite took {elapsed:.2f}s")
        
        # Extract text from response
        raw_text = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            # modern responses include message.content
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                raw_text = choice.message.content.strip()
            elif hasattr(choice, 'text'):
                raw_text = choice.text.strip()
        else:
            raw_text = str(response).strip()
        
        logger.debug(f"LLM rewrite raw output: {raw_text}")
        
        # Parse JSON response with multiple strategies
        queries = _parse_llm_rewrite_response(raw_text, max_variants)
        
        if queries:
            # Clean and normalize each query
            cleaned = [rewrite_query_simple(q) for q in queries if q and q.strip()]
            # Remove duplicates while preserving order
            unique = []
            seen = set()
            for q in cleaned:
                if q and q not in seen:
                    unique.append(q)
                    seen.add(q)
            
            if unique:
                logger.info(f"LLM generated {len(unique)} rewrite variants")
                return unique[:max_variants]
        
        # If parsing failed, fall back
        logger.warning("LLM rewrite parsing failed, using fallback")
        
    except OpenAIError as e:
        logger.error(f"OpenAI API error during rewrite: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during LLM rewrite: {e}", exc_info=True)
    
    # Fallback: return deterministic rewrite
    return [rewrite_query_simple(user_question)]


def _parse_llm_rewrite_response(raw_text: str, max_variants: int) -> List[str]:
    """
    Robust parsing of LLM rewrite response with multiple fallback strategies.
    Returns list of query strings.
    """
    # Strategy 1: Parse as JSON
    try:
        # Remove markdown code blocks if present
        text = re.sub(r'^```json?\s*', '', raw_text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        
        parsed = json.loads(text)
        
        # Handle different JSON structures
        if isinstance(parsed, dict):
            queries = parsed.get("queries", parsed.get("rewrites", []))
            if isinstance(queries, list):
                return [str(q) for q in queries if q]
        elif isinstance(parsed, list):
            return [str(q) for q in parsed if q]
    
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract array-like structure with regex
    try:
        array_match = re.search(r'\[(.*?)\]', raw_text, re.DOTALL)
        if array_match:
            content = array_match.group(1)
            # Extract quoted strings
            queries = re.findall(r'["\']([^"\']+)["\']', content)
            if queries:
                return queries
    except Exception:
        pass
    
    # Strategy 3: Split by lines and clean
    try:
        lines = [
            line.strip(' -•*"\'\t\r\n')
            for line in raw_text.splitlines()
            if line.strip()
        ]
        
        # Filter reasonable-looking queries
        candidates = []
        for line in lines:
            # Skip lines that look like instructions or metadata
            if any(skip in line.lower() for skip in ['query', 'rewrite', 'output', 'format']):
                continue
            # Keep lines that look like search queries
            word_count = len(line.split())
            if 2 <= word_count <= 15 and len(line) >= 3:
                candidates.append(line)
        
        if candidates:
            return candidates[:max_variants]
    
    except Exception:
        pass
    
    # Strategy 4: Comma or newline separated
    try:
        parts = re.split(r'[,\n]+', raw_text)
        cleaned = [
            p.strip(' -•*"\'\t\r\n')
            for p in parts
            if p.strip() and 2 <= len(p.split()) <= 15
        ]
        if cleaned:
            return cleaned[:max_variants]
    except Exception:
        pass
    
    # All strategies failed
    return []


# ========================
# Embedding Cache
# ========================
if Config.ENABLE_EMBEDDING_CACHE:
    @lru_cache(maxsize=Config.CACHE_MAX_SIZE)
    def embed_cached(text: str) -> List[float]:
        """Cached embedding function"""
        return embed(text)
    
    logger.info(f"Embedding cache enabled (max size: {Config.CACHE_MAX_SIZE})")
else:
    def embed_cached(text: str) -> List[float]:
        """Non-cached embedding function"""
        return embed(text)
    
    logger.info("Embedding cache disabled")


# ========================
# Qdrant Vector Search
# ========================
def build_qdrant_filter(filters: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Build Qdrant filter from extracted filters.
    Supports basic equality matching.
    """
    if not filters:
        return None
    
    must_clauses = []
    
    # Map filter keys to Qdrant payload fields
    filter_mapping = {
        "source": "source_file",
        "type": "doc_type",
        "date": "date",
    }
    
    for filter_key, filter_value in filters.items():
        if filter_key in filter_mapping:
            payload_key = filter_mapping[filter_key]
            must_clauses.append({
                "key": payload_key,
                "match": {"value": filter_value}
            })
    
    if must_clauses:
        return {"must": must_clauses}
    
    return None


def qdrant_search(
    query_text: str,
    top_k: int = Config.TOP_K,
    filters: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Execute vector search on Qdrant via REST API.
    Supports optional metadata filters.
    """
    # Get embedding (with caching if enabled)
    query_vector = embed_cached(query_text)
    
    # Build request payload
   
   
    payload = {
        "vector": query_vector,  # ✅ Changed from nested dict
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }

       
    # Add filter if provided
    qdrant_filter = build_qdrant_filter(filters)
    if qdrant_filter:
        payload["filter"] = qdrant_filter
        logger.debug(f"Using Qdrant filter: {qdrant_filter}")
    
    url = f"{Config.QDRANT_URL}/collections/{Config.COLLECTION}/points/search"

    try:
        # Use session with connection pooling
        session = get_http_session()
        response = session.post(
            url,
            headers={
                "api-key": Config.QDRANT_API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=Config.QDRANT_TIMEOUT
        )
        response.raise_for_status()
        resp_json = response.json()

        # If no results were returned, log the raw payload to help debug
        # embedding/model mismatches (e.g., switching from Anthropic → OpenAI).
        hits = extract_search_results(resp_json)
        if not hits:
            logger.debug(f"Qdrant returned no hits. Raw response: {json.dumps(resp_json)})")

        return resp_json
    
    except requests.Timeout:
        logger.error(f"Qdrant search timeout after {Config.QDRANT_TIMEOUT}s")
        raise
    except requests.HTTPError as e:
        logger.error(f"Qdrant HTTP error: {e.response.status_code} - {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Qdrant search error: {e}", exc_info=True)
        raise


def extract_search_results(qdrant_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize Qdrant response structure to unified hit format.
    Handles different response schemas.
    """
    # Try multiple common response structures
    if "result" in qdrant_response:
        result = qdrant_response["result"]
        
        if isinstance(result, list):
            return result
        elif isinstance(result, dict):
            # Try common nested keys
            for key in ["points", "hits", "results"]:
                if key in result:
                    return result[key]
            return [result]  # Single result wrapped in dict
    
    # Fallback to top-level hits
    if "hits" in qdrant_response:
        return qdrant_response["hits"]
    
    logger.warning("Unexpected Qdrant response structure")
    return []


# ========================
# Retrieval Orchestration
# ========================
def retrieve(
    user_question: str,
    rewritten_queries: List[str],
    filters: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve candidates using multiple query variants.
    - Runs vector search for each rewritten query (parallel if enabled)
    - Deduplicates results
    - Adds BM25 keyword scores
    - Computes hybrid scores
    """
    all_hits: List[Dict[str, Any]] = []
    seen_ids = set()
    
    if Config.ENABLE_PARALLEL_RETRIEVAL and len(rewritten_queries) > 1:
        # Parallel retrieval
        logger.info(f"Running parallel retrieval for {len(rewritten_queries)} query variants")
        
        def fetch_query(query_data: Tuple[int, str]) -> Tuple[int, str, List[Dict[str, Any]]]:
            """Helper function for parallel execution"""
            idx, query = query_data
            try:
                qdrant_response = qdrant_search(
                    query,
                    top_k=Config.TOP_K_EXPAND,
                    filters=filters
                )
                hits = extract_search_results(qdrant_response)
                logger.debug(f"Query variant {idx+1} returned {len(hits)} hits")
                return idx, query, hits
            except Exception as e:
                logger.error(f"Retrieval failed for query variant '{query}': {e}")
                return idx, query, []
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=Config.MAX_PARALLEL_WORKERS) as executor:
            query_data = list(enumerate(rewritten_queries))
            future_to_query = {
                executor.submit(fetch_query, qd): qd for qd in query_data
            }
            
            # Collect results as they complete
            results = []
            for future in as_completed(future_to_query):
                try:
                    idx, query, hits = future.result()
                    results.append((idx, query, hits))
                except Exception as e:
                    logger.error(f"Parallel retrieval task failed: {e}")
            
            # Sort results by original query order
            results.sort(key=lambda x: x[0])
            
            # Deduplicate hits
            for idx, query, hits in results:
                logger.info(f"Processing query variant {idx+1}/{len(rewritten_queries)}: {query}")
                for hit in hits:
                    hit_id = _extract_hit_id(hit)
                    if hit_id not in seen_ids:
                        seen_ids.add(hit_id)
                        all_hits.append(hit)
    else:
        # Sequential retrieval (fallback or single query)
        for idx, query in enumerate(rewritten_queries):
            logger.info(f"Retrieving with query variant {idx+1}/{len(rewritten_queries)}: {query}")
            
            try:
                qdrant_response = qdrant_search(
                    query,
                    top_k=Config.TOP_K_EXPAND,
                    filters=filters
                )
                hits = extract_search_results(qdrant_response)
                
                # Deduplicate by ID
                for hit in hits:
                    hit_id = _extract_hit_id(hit)
                    
                    if hit_id not in seen_ids:
                        seen_ids.add(hit_id)
                        all_hits.append(hit)
                
                logger.debug(f"Query variant {idx+1} returned {len(hits)} hits")
            
            except Exception as e:
                logger.error(f"Retrieval failed for query variant '{query}': {e}")
                # Continue with other variants
    
    if not all_hits:
        logger.warning("No results retrieved from any query variant")
        return []
    
    logger.info(f"Total unique hits retrieved: {len(all_hits)}")
    
    # Add BM25 keyword scores
    try:
        all_hits = run_keyword_search(user_question, all_hits)
        logger.debug("BM25 scores added successfully")
    except Exception as e:
        logger.warning(f"BM25 scoring failed: {e}")
        # Add default BM25 score if failed
        for hit in all_hits:
            if "bm25_score" not in hit:
                hit["bm25_score"] = 0.0
    
    # Compute hybrid scores
    for hit in all_hits:
        vector_score = float(hit.get("score", 0.0))
        bm25_score = float(hit.get("bm25_score", 0.0))
        
        # Normalize BM25 score (diminishing returns)
        normalized_bm25 = bm25_score / (1 + bm25_score)
        
        # Weighted combination
        hit["hybrid_score"] = (
            Config.VECTOR_WEIGHT * vector_score +
            Config.BM25_WEIGHT * normalized_bm25
        )
    
    # Sort by hybrid score (descending)
    all_hits.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
    
    return all_hits


def _extract_hit_id(hit: Dict[str, Any]) -> str:
    """
    Extract unique ID from hit, handling different response structures.
    Falls back to hash of payload if no ID found.
    """
    # Try common ID locations
    if "id" in hit:
        return str(hit["id"])
    
    if "point" in hit and isinstance(hit["point"], dict):
        if "id" in hit["point"]:
            return str(hit["point"]["id"])
    
    # Fallback: create ID from payload
    payload = hit.get("payload", {})
    source = payload.get("source_file", "")
    chunk = payload.get("chunk_index", "")
    return f"{source}#{chunk}" if source else str(hash(json.dumps(payload, sort_keys=True)))


# ========================
# Reranking
# ========================
def rerank_hits(user_question: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply optional reranking model.
    Limits candidates to reasonable size for LLM-based rerankers.
    """
    if not RERANK_ENABLED:
        logger.debug("Reranking disabled")
        return hits
    
    if not hits:
        return hits
    
    # Limit reranking to top candidates (performance optimization)
    MAX_RERANK_CANDIDATES = 20
    candidates = hits[:MAX_RERANK_CANDIDATES]
    
    try:
        logger.info(f"Reranking {len(candidates)} candidates")
        reranked = rerank(user_question, candidates)
        
        # Append remaining hits that weren't reranked
        if len(hits) > MAX_RERANK_CANDIDATES:
            reranked.extend(hits[MAX_RERANK_CANDIDATES:])
        
        return reranked
    
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return hits


# ========================
# Context Building
# ========================

# Initialize tokenizer for accurate token counting
_tokenizer_cache = {}

def get_tokenizer(model: str = Config.OPENAI_MODEL):
    """Get or create cached tokenizer for the model"""
    if model not in _tokenizer_cache:
        if TIKTOKEN_AVAILABLE:
            try:
                # Use cl100k_base encoding (compatible with many OpenAI/GPT models)
                encoding = tiktoken.get_encoding("cl100k_base")
                _tokenizer_cache[model] = encoding
                logger.debug(f"Initialized tiktoken encoder for {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}")
                _tokenizer_cache[model] = None
        else:
            _tokenizer_cache[model] = None
    
    return _tokenizer_cache[model]


def count_tokens(text: str) -> int:
    """
    Accurately count tokens using tiktoken.
    Falls back to character-based estimation if tiktoken unavailable.
    """
    tokenizer = get_tokenizer()
    
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using fallback")
    
    # Fallback to character-based estimation
    return int(len(text) / Config.CHARS_PER_TOKEN_ESTIMATE)


def estimate_tokens(text: str) -> int:
    """Alias for count_tokens for backward compatibility"""
    return count_tokens(text)


def build_context_and_sources(
    hits: List[Dict[str, Any]],
    max_tokens: int = Config.MAX_CONTEXT_TOKENS
) -> Tuple[str, List[str], int]:
    """
    Build context string and source list from retrieved hits.
    Token-aware truncation to avoid context window overflow.
    Returns: (context_string, source_list, token_count)
    """
    context_parts: List[str] = []
    sources: List[str] = []
    total_tokens = 0

    for idx, hit in enumerate(hits):
        payload = hit.get("payload", {}) or {}

        # Extract text content
        text = payload.get("text", "") or payload.get("content", "")
        if not text:
            continue

        # Extract metadata
        score = float(hit.get("hybrid_score", hit.get("score", 0)))
        source_file = payload.get("source_file") or str(hit.get("id", f"unknown_{idx}"))
        chunk_index = payload.get("chunk_index")

        # Build source reference
        source_ref = (
            f"{source_file}#chunk{chunk_index}"
            if chunk_index is not None
            else source_file
        )

        # Estimate tokens for this chunk
        chunk_tokens = estimate_tokens(text)

        # Check if adding this chunk would exceed limit
        if total_tokens + chunk_tokens > max_tokens:
            logger.info(
                f"Context token limit reached ({total_tokens}/{max_tokens}). "
                f"Using {len(context_parts)} chunks."
            )
            break

        # Format context entry with clear structure
        # Rank helps LLM prioritize higher-relevance content
        context_entry = (
            f"[DOCUMENT {idx + 1}] Source: {source_ref} | Relevance: {score:.2f}\n"
            f"{text}\n"
            f"[END DOCUMENT {idx + 1}]"
        )

        context_parts.append(context_entry)
        sources.append(source_ref)
        total_tokens += chunk_tokens

    context = "\n\n".join(context_parts)

    logger.info(
        f"Built context from {len(sources)} chunks, "
        f"~{total_tokens} tokens (~{len(context)} chars)"
    )

    return context, sources, total_tokens


# ========================
# LLM Answer Generation
# ========================
def call_openai_chat_api(
    user_question: str,
    context: str,
    system_prompt: str
) -> str:
    """
    Generate answer using OpenAI hosted chat API.
    Note: If context is already embedded in system_prompt (from build_prompt),
    pass empty string for context parameter.
    """
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # If context is provided separately (legacy behavior), include it in message
    if context and context.strip():
        user_message = f"""RETRIEVED CONTEXT:
{context}

USER QUESTION:
{user_question}

INSTRUCTIONS:
- Answer the question using ONLY information from the retrieved context above
- If the context doesn't contain enough information, say so clearly
- Cite specific sources when making claims
- Be concise and direct
"""
    else:
        # Context is already in system_prompt, just send the question
        user_message = user_question
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=Config.MAX_COMPLETION_TOKENS,
            timeout=Config.OPENAI_TIMEOUT
        )
        
        # Extract text from response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return choice.message.content.strip()
            elif hasattr(choice, 'text'):
                return choice.text.strip()
        
        return str(response).strip()
    
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI API: {e}", exc_info=True)
        raise


# ========================
# Fallback Response Generators
# ========================
def _generate_llm_failure_fallback(context: str, question: str, intents: List[str], conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generate fallback response when LLM API fails.
    Uses retrieved context to provide a simple answer.
    """
    if not context:
        return _generate_fallback_response(intents, question, conversation_history)

    # Extract key information from context
    context_preview = context[:500] + "..." if len(context) > 500 else context

    return (
        f"I found relevant information but encountered a temporary issue generating a complete response. "
        f"Here's what I found:\n\n{context_preview}\n\n"
        f"Please try asking your question again, or contact support@zevpoint.com for immediate assistance."
    )


def _generate_fallback_response(intents: List[str], question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generate a helpful fallback response when no information is found.
    Uses intent and conversation context to provide appropriate message.
    """
    # Import here to avoid circular imports
    from shared.prompt_manager import extract_conversation_context

    # Get conversation context to avoid asking questions we already know
    ctx = extract_conversation_context(conversation_history) if conversation_history else {}

    if "sales" in intents:
        # If we already know the vehicle, don't ask again
        if ctx.get("vehicle"):
            return (
                "I couldn't find specific details for that. "
                "Could you tell me more about what you're looking for?"
            )
        return (
            "I don't have specific information about that in my knowledge base. "
            "However, I'd be happy to help you find the right EV charger! "
            "Could you tell me which electric vehicle you drive? "
            "That will help me recommend the best charging solution for you."
        )

    if "agent_handoff" in intents:
        return (
            "I understand you'd like to speak with our support team. "
            "You can reach us at support@zevpoint.com or call us during business hours. "
            "Is there anything specific I can help you with in the meantime?"
        )

    if "service" in intents:
        return (
            "I don't have specific information about that service query. "
            "For technical support or service-related questions, please contact our support team at support@zevpoint.com. "
            "They'll be able to assist you with troubleshooting and maintenance."
        )

    # Default fallback
    return (
        "I couldn't find specific information to answer your question. "
        "Could you rephrase your question or provide more details? "
        "I'm here to help with EV chargers, installation, pricing, and technical specifications."
    )


def _parse_answer_nudge(llm_response: str) -> tuple:
    """
    Parse the LLM response to extract ANSWER and NUDGE parts.

    Expected format:
    ANSWER: [response text]
    NUDGE: [follow-up text or "none"]

    Also handles cases where LLM doesn't use ANSWER: prefix but includes NUDGE:

    Returns:
        tuple: (answer_text, nudge_text or None)
    """
    answer = llm_response
    nudge = None

    # Check if NUDGE: is present anywhere in the response
    if "NUDGE:" in llm_response:
        parts = llm_response.split("NUDGE:", 1)

        # Extract answer part (everything before NUDGE:)
        answer_part = parts[0].strip()

        # Remove ANSWER: prefix if present
        if answer_part.startswith("ANSWER:"):
            answer = answer_part[7:].strip()  # Remove "ANSWER:" (7 chars)
        else:
            answer = answer_part

        # Extract nudge part
        nudge_text = parts[1].strip()
        # Check if nudge is "none" or empty
        if nudge_text.lower() not in ("none", ""):
            nudge = nudge_text

    elif "ANSWER:" in llm_response:
        # Only ANSWER: present, no NUDGE:
        answer = llm_response.split("ANSWER:", 1)[1].strip()

    return answer, nudge


# ========================
# Main Pipeline Orchestrator
# ========================
def answer_question(
    user_question: str,
    conversation_history: List[Dict[str, str]] = None,
    top_k: int = Config.TOP_K,
    enable_llm_rewrite: bool = False
) -> PipelineResult:
    """Main RAG pipeline orchestrator."""
    pipeline_start = time.time()
    metrics = QueryMetrics()
    
    logger.info("=" * 60)
    logger.info("RAG PIPELINE START")
    logger.info("=" * 60)
    logger.info(f"Question: {user_question}")
    
    try:
        # 1. Validate input
        user_question = validate_question(user_question)
        
        # 2. Detect intents EARLY (before any error returns)
        intents = detect_intent(user_question)
        logger.info(f"Detected intents: {intents}")
        
        # 3. Preprocess: extract filters + normalize
        preprocess_start = time.time()
        clean_query, filters = extract_filters(user_question)

        # Enrich vague queries with context from conversation history
        # This helps retrieval for follow-up questions like "what's the price"
        enriched_query = enrich_query_with_context(clean_query, conversation_history)

        normalized_query = normalize_query(enriched_query)
        deterministic_rewrite = rewrite_query_simple(normalized_query)
        metrics.preprocessing_time = time.time() - preprocess_start

        logger.info(f"Original query: {clean_query}")
        if enriched_query != clean_query:
            logger.info(f"Enriched query: {enriched_query}")
        logger.info(f"Deterministic rewrite: {deterministic_rewrite}")
        if filters:
            logger.info(f"Filters: {filters}")
        
        # 4. Optionally generate LLM rewrites
        rewritten_queries = [deterministic_rewrite]
        
        if enable_llm_rewrite:
            llm_rewrite_start = time.time()
            try:
                llm_rewrites = rewrite_query_with_llm(
                    user_question,
                    max_variants=Config.MAX_REWRITE_VARIANTS
                )
                
                for rq in llm_rewrites:
                    if rq not in rewritten_queries:
                        rewritten_queries.append(rq)
                
                logger.info(f"LLM rewrites: {llm_rewrites}")
            
            except Exception as e:
                logger.warning(f"LLM rewrite failed, using deterministic only: {e}")
            
            metrics.llm_rewrite_time = time.time() - llm_rewrite_start
        
        logger.info(f"Final query variants: {rewritten_queries}")
        
        # 5. Retrieve candidates
        retrieval_start = time.time()
        all_hits = retrieve(user_question, rewritten_queries, filters=filters)
        metrics.retrieval_time = time.time() - retrieval_start
        metrics.search_time_ms = metrics.retrieval_time * 1000
        metrics.retrieved_count = len(all_hits)
        
        if not all_hits:
            logger.warning("No results retrieved from vector search - continuing with LLM only")

        # 6. Rerank
        rerank_start = time.time()
        reranked_hits = rerank_hits(user_question, all_hits)
        metrics.rerank_time = time.time() - rerank_start
        metrics.reranked_count = len(reranked_hits)
        
        # 7. Select top-k and build context
        top_hits = reranked_hits[:top_k]
        context, sources, context_tokens = build_context_and_sources(
            top_hits,
            max_tokens=Config.MAX_CONTEXT_TOKENS
        )
        
        metrics.chunks_used = len(top_hits)
        metrics.context_tokens = context_tokens
        
        if not context:
            logger.warning("No context could be built from results - will use conversation history only")

        # 8. Generate answer with conversation history
        generation_start = time.time()
        
        # Build dynamic system prompt with context
        system_prompt = build_prompt(
            intents=intents,
            question=user_question,
            context=context,
            conversation_history=conversation_history
        )
        
        logger.info("Using dynamic system prompt with RAG context")
        
        # Build messages array
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided (last 6 messages = 3 exchanges)
        if conversation_history:
            for msg in conversation_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            logger.info(f"Including {len(conversation_history[-6:])} messages from history")
        
        # Add current question
        messages.append({
            "role": "user",
            "content": user_question
        })
        
        # Call OpenAI API with retry logic
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        max_retries = 2
        answer_text = None

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=messages,
                    max_completion_tokens=Config.MAX_COMPLETION_TOKENS,
                    timeout=Config.OPENAI_TIMEOUT
                )

                # Extract answer
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        answer_text = choice.message.content.strip()
                    elif hasattr(choice, 'text'):
                        answer_text = choice.text.strip()
                    else:
                        answer_text = str(choice).strip()
                else:
                    answer_text = str(response).strip()

                # Success - break retry loop
                break

            except OpenAIError as e:
                logger.error(f"OpenAI API error (attempt {attempt + 1}/{max_retries + 1}): {e}")

                if attempt == max_retries:
                    # Final attempt failed - use fallback
                    logger.error("All OpenAI API attempts failed, using fallback response")
                    answer_text = _generate_llm_failure_fallback(context, user_question, intents, conversation_history)
                    break

                # Wait before retry (exponential backoff)
                time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {e}", exc_info=True)

                if attempt == max_retries:
                    # Final attempt failed - use fallback
                    answer_text = _generate_llm_failure_fallback(context, user_question, intents, conversation_history)
                    break

                time.sleep(2 ** attempt)
        
        metrics.llm_generation_time = time.time() - generation_start
        metrics.total_time = time.time() - pipeline_start

        # Parse ANSWER and NUDGE from LLM response
        answer_text, nudge_text = _parse_answer_nudge(answer_text)
        if nudge_text:
            logger.info(f"Parsed nudge: {nudge_text[:50]}...")

        # Estimate confidence
        top_score = top_hits[0].get("hybrid_score", 0.0) if top_hits else 0.0
        confidence = min(top_score, 1.0)

        logger.info("=" * 60)
        logger.info("RAG PIPELINE COMPLETE")
        logger.info(f"Total time: {metrics.total_time:.2f}s")
        logger.info(f"Retrieved: {metrics.retrieved_count}, Used: {metrics.chunks_used}")
        logger.info(f"Confidence: {confidence:.2f}")
        logger.info("=" * 60)

        return PipelineResult(
            answer=answer_text,
            sources=sources,
            metrics=metrics,
            filters=filters,
            rewritten_queries=rewritten_queries,
            retrieved_chunks=[hit.get("payload", {}).get("text", "")[:200] for hit in top_hits],
            intents=intents,
            confidence_score=confidence,
            search_time_ms=metrics.search_time_ms,
            llm_time_ms=metrics.llm_generation_time * 1000,
            nudge=nudge_text
        )
    
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        metrics.total_time = time.time() - pipeline_start
        
        return PipelineResult(
            answer=f"An error occurred while processing your question: {str(e)}",
            sources=[],
            metrics=metrics,
            filters={},
            rewritten_queries=[],
            retrieved_chunks=[],
            intents=[],  # ✅ Return empty list instead of undefined variable
            confidence_score=0.0,
            search_time_ms=metrics.search_time_ms,
            llm_time_ms=metrics.llm_generation_time * 1000
        )

# ========================
# CLI Entry Point
# ========================
def main():
    """Command-line interface for testing"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline Query Tool")
    parser.add_argument(
        "question",
        nargs="+",
        help="Question to answer"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=Config.TOP_K,
        help=f"Number of chunks to use (default: {Config.TOP_K})"
    )
    parser.add_argument(
        "--llm-rewrite",
        action="store_true",
        help="Enable LLM-based query rewriting"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    # Join question parts
    question = " ".join(args.question)
    
    try:
        # Run pipeline
        result = answer_question(
            question,
            top_k=args.top_k,
            enable_llm_rewrite=args.llm_rewrite
        )
        
        if args.json:
            # Output as JSON
            print(json.dumps(result.to_dict(), indent=2))
        else:
            # Human-readable output
            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result.answer)
            print("\n" + "=" * 60)
            print("SOURCES")
            print("=" * 60)
            for i, source in enumerate(result.sources, 1):
                print(f"{i}. {source}")
            
            print("\n" + "=" * 60)
            print("METRICS")
            print("=" * 60)
            print(f"Total time: {result.metrics.total_time:.2f}s")
            print(f"  - Preprocessing: {result.metrics.preprocessing_time:.2f}s")
            print(f"  - LLM rewrite: {result.metrics.llm_rewrite_time:.2f}s")
            print(f"  - Retrieval: {result.metrics.retrieval_time:.2f}s")
            print(f"  - Reranking: {result.metrics.rerank_time:.2f}s")
            print(f"  - Generation: {result.metrics.llm_generation_time:.2f}s")
            print(f"Retrieved: {result.metrics.retrieved_count}")
            print(f"Chunks used: {result.metrics.chunks_used}")
            print(f"Context tokens: {result.metrics.context_tokens}")
            print(f"Confidence: {result.confidence_score:.2f}")
            
            if result.filters:
                print(f"Filters applied: {result.filters}")
            
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()