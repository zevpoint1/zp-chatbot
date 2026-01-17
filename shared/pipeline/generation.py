"""
Generation module - LLM calls, context building, response parsing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI, OpenAIError

from shared.config import AppConfig
from shared.pipeline.config import Config

logger = logging.getLogger(__name__)


# ========================
# Token Counting
# ========================
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available, falling back to character-based token estimation")

_tokenizer_cache = {}


def get_tokenizer(model: str = None):
    """Get or create cached tokenizer for the model"""
    if model is None:
        model = Config.OPENAI_MODEL

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


# ========================
# Context Building
# ========================
def build_context_and_sources(
    hits: List[Dict[str, Any]],
    max_tokens: int = None
) -> Tuple[str, List[str], int]:
    """
    Build context string and source list from retrieved hits.
    Token-aware truncation to avoid context window overflow.
    Returns: (context_string, source_list, token_count)
    """
    if max_tokens is None:
        max_tokens = Config.MAX_CONTEXT_TOKENS

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
# LLM API Call
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
# Response Parsing
# ========================
def parse_answer_nudge(llm_response: str) -> Tuple[str, Optional[str]]:
    """
    Parse the LLM response to extract ANSWER and NUDGE parts.

    Expected format:
    ANSWER: [response text]
    NUDGE: [follow-up text or "none"]

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
# Fallback Responses
# ========================
def generate_llm_failure_fallback(
    context: str,
    question: str,
    intents: List[str],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate fallback response when LLM API fails.
    Uses retrieved context to provide a simple answer.
    """
    if not context:
        return generate_fallback_response(intents, question, conversation_history)

    # Extract key information from context
    preview_len = AppConfig.CONTEXT_PREVIEW_LENGTH
    context_preview = context[:preview_len] + "..." if len(context) > preview_len else context

    return (
        f"I found relevant information but encountered a temporary issue generating a complete response. "
        f"Here's what I found:\n\n{context_preview}\n\n"
        f"Please try asking your question again, or contact support@zevpoint.com for immediate assistance."
    )


def generate_fallback_response(
    intents: List[str],
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
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
