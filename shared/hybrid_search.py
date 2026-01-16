import re
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    """Simple tokenizer for BM25"""
    return re.findall(r"\w+", text.lower())

def run_keyword_search(query: str, hits: list, top_k: int = 10):
    """
    Apply BM25 keyword scoring to the result set returned by Qdrant vector search.
    """
    docs = [h["payload"].get("text", "") for h in hits]
    tokenized_docs = [tokenize(d) for d in docs]

    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)  # list of floats

    # Attach BM25 score to each hit
    for hit, score in zip(hits, scores):
        hit["bm25_score"] = float(score)

    return hits
