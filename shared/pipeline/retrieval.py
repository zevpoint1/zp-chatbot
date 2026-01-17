"""
Retrieval module - Qdrant search, BM25, hybrid scoring, reranking.
"""

import json
import logging
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from shared.config import AppConfig
from shared.pipeline.config import Config
from shared.embed_text import embed
from shared.hybrid_search import run_keyword_search

logger = logging.getLogger(__name__)


# ========================
# Optional Reranker
# ========================
try:
    from shared.reranker import rerank
    RERANK_ENABLED = True
except Exception as e:
    logging.warning(f"Reranker not available: {e}")
    RERANK_ENABLED = False


# ========================
# Connection Pool
# ========================
_http_session = None


def get_http_session():
    """Get or create HTTP session with connection pooling"""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=AppConfig.HTTP_POOL_CONNECTIONS,
            pool_maxsize=AppConfig.HTTP_POOL_MAXSIZE,
            max_retries=AppConfig.HTTP_MAX_RETRIES,
            pool_block=False
        )
        _http_session.mount('http://', adapter)
        _http_session.mount('https://', adapter)
        logger.info("HTTP session with connection pooling initialized")
    return _http_session


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
# Qdrant Filter Builder
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


# ========================
# Qdrant Vector Search
# ========================
def qdrant_search(
    query_text: str,
    top_k: int = None,
    filters: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Execute vector search on Qdrant via REST API.
    Supports optional metadata filters.
    """
    if top_k is None:
        top_k = Config.TOP_K

    # Get embedding (with caching if enabled)
    query_vector = embed_cached(query_text)

    # Build request payload
    payload = {
        "vector": query_vector,
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

        # If no results were returned, log for debugging
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
    candidates = hits[:AppConfig.MAX_RERANK_CANDIDATES]

    try:
        logger.info(f"Reranking {len(candidates)} candidates")
        reranked = rerank(user_question, candidates)

        # Append remaining hits that weren't reranked
        if len(hits) > AppConfig.MAX_RERANK_CANDIDATES:
            reranked.extend(hits[AppConfig.MAX_RERANK_CANDIDATES:])

        return reranked

    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return hits
