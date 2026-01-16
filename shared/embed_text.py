"""
embed_text.py - Production-ready embedding service
REMOVED: dotenv loading (use Azure Application Settings instead)
"""

import os
import json
import base64
import hashlib
import logging
from functools import lru_cache
from typing import List
# Prefer OpenAI hosted embeddings; fall back to Azure if configured
try:
    from openai import OpenAI, OpenAIError
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OpenAIError = Exception
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding model name (OpenAI or Azure)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Azure environment variables (optional fallback)
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-ada-002")

# Initialize client depending on available configuration
client = None
if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized for embeddings")
elif AZURE_ENDPOINT and AZURE_KEY:
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version="2023-05-15"
        )
        logger.info("Azure OpenAI client initialized for embeddings")
    except Exception:
        logger.error("Failed to initialize Azure OpenAI client for embeddings")
else:
    logger.error("Missing OpenAI or Azure OpenAI environment variables for embeddings!")

# Embedding dimension defaults (may vary by model)
# Map common embedding model names to their expected vector dimension
_MODEL_DIM_MAP = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

# Determine dimension from model name; default to 1536 for safety
EMBEDDING_DIMENSION = _MODEL_DIM_MAP.get(EMBEDDING_MODEL, 1536)
if EMBEDDING_MODEL not in _MODEL_DIM_MAP:
    logger.warning(f"Unknown embedding model '{EMBEDDING_MODEL}', defaulting dimension to {EMBEDDING_DIMENSION}")


# ------------------------------
# Utility: hashing key
# ------------------------------
def _key_for_text(text: str) -> str:
    return f"embed:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


# ------------------------------
# Core Azure Embedding Call
# ------------------------------
def _embed_remote(text: str) -> List[float]:
    """
    Calls the configured embedding service (OpenAI or Azure) for embedding.
    """

    if client is None:
        logger.error("No embedding client configured")
        raise RuntimeError("No embedding client configured")

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )

        embedding = response.data[0].embedding

        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMENSION}, got {len(embedding)}"
            )


        return embedding

    except OpenAIError as e:
        logger.error(f"OpenAI embedding API error: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected embedding error: {e}")
        raise


# ------------------------------
# Local LRU Cache
# ------------------------------
@lru_cache(maxsize=2048)
def _embed_cached_local(text: str) -> List[float]:
    return _embed_remote(text)


# ------------------------------
# Public: embed()
# ------------------------------
def embed(text: str) -> List[float]:
    """
    Embeds a single text string.
    Uses:
      - Redis cache (if enabled)
      - Local LRU cache
    """

    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIMENSION

    text = text.strip()
    cache_key = _key_for_text(text)

  

    # 2. Use local LRU + remote embedding
    vec = _embed_cached_local(text)

    return vec


# ------------------------------
# Batch Embedding
# ------------------------------
def embed_batch(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Embeds multiple texts efficiently with batching.
    Falls back to single embeddings if a batch fails.
    """

    if not texts:
        return []

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )

            batch_vecs = [item.embedding for item in response.data]
            results.extend(batch_vecs)

            logger.info(f"Embedded batch {i//batch_size + 1} ({len(batch)} items)")

        except Exception as e:
            logger.error(f"Batch embedding failed — falling back: {e}")

            # Fallback: embed one by one
            for text in batch:
                try:
                    results.append(embed(text))
                except Exception as sub_e:
                    logger.error(f"Individual embedding failed: {sub_e}")
                    results.append([0.0] * EMBEDDING_DIMENSION)

    return results


# ------------------------------
# Manual Test
# ------------------------------
if __name__ == "__main__":
    test = "Hello world, this is a test"
    vec = embed(test)
    print("Single embedding length:", len(vec))
    print("First 5 values:", vec[:5])

    batch_vecs = embed_batch(["one", "two", "three"])
    print("\nBatch embeddings:", len(batch_vecs))
    print("Vector size:", len(batch_vecs[0]))
