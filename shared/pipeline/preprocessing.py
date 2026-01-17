"""
Query preprocessing - validation, normalization, rewriting, context extraction.
"""

import re
import json
import time
import logging
from typing import List, Dict, Optional, Tuple

from openai import OpenAI, OpenAIError

from shared.config import AppConfig
from shared.pipeline.config import Config
from shared.pipeline.models import ValidationError

logger = logging.getLogger(__name__)


# ========================
# Synonym Map
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
}


# ========================
# Product and Vehicle Patterns
# ========================
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


# ========================
# Compiled Patterns
# ========================
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


# ========================
# Input Validation
# ========================
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
# Query Normalization
# ========================
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


# ========================
# Filter Extraction
# ========================
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


# ========================
# Query Rewriting
# ========================
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


def rewrite_query_with_llm(
    user_question: str,
    max_variants: int = None
) -> List[str]:
    """
    Use OpenAI hosted chat API to generate optimized retrieval queries.
    Returns list of rewritten queries with robust parsing and fallbacks.
    """
    if max_variants is None:
        max_variants = Config.MAX_REWRITE_VARIANTS

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
# Context Extraction
# ========================
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
    for msg in reversed(conversation_history[-AppConfig.CONVERSATION_HISTORY_LIMIT:]):
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
        len(query_words) <= AppConfig.VAGUE_QUERY_MAX_WORDS and
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
