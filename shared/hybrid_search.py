import re
from rank_bm25 import BM25Okapi

def tokenize(text: str):
    """
    Enhanced tokenizer for BM25 that preserves product specs.
    Handles: "7.5kW" -> "7.5kw", "Rs. 20,999" -> "rs", "20999", "32A" -> "32a"
    """
    text = text.lower()
    # Keep numbers with units together (e.g., 7.5kw, 32a, 16a)
    text = re.sub(r'(\d+\.?\d*)\s*(kw|kva|kwh|amp|a|v|w|mm|cm|m|kg|rs|inr)', r'\1\2', text)
    # Remove commas from numbers (20,999 -> 20999)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Tokenize: alphanumeric sequences including decimals
    tokens = re.findall(r'[\w\.]+', text)
    # Filter out standalone dots
    return [t.strip('.') for t in tokens if t.strip('.')]

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
