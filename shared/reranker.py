# reranker.py
import os
import cohere
import logging

logger = logging.getLogger(__name__)

COHERE_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_KEY:
    logger.warning("COHERE_API_KEY missing – reranking disabled")
    cohere_client = None
else:
    cohere_client = cohere.Client(COHERE_KEY)


def rerank(query: str, hits):
    """
    Rerank retrieved chunks using Cohere Rerank API.
    Each hit must contain payload['text'].
    """

    if not hits:
        return hits

    if not cohere_client:
        logger.warning("Cohere not configured – skipping rerank")
        return hits

    # Extract document text from hits
    documents = [h.get("payload", {}).get("text", "") for h in hits]

    try:
        response = cohere_client.rerank(
            query=query,
            documents=documents,
            top_n=len(documents),
            model="rerank-english-v3.0"
        )

        # Attach rerank score back to hits
        for new_item, hit in zip(response.results, hits):
            hit["rerank_score"] = new_item.relevance_score

        # Sort highest → lowest
        hits = sorted(hits, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        return hits

    except Exception as e:
        logger.error(f"Rerank failed: {e}")
        return hits
