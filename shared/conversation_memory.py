"""
Dynamic conversation memory extraction using LLM.

Instead of hardcoding field extraction (vehicle, preference, three-phase, etc.),
this module uses an LLM to:
1. Extract key facts from conversation history
2. Identify what questions have been asked and answered
3. Generate dynamic rules about what NOT to ask again
"""

import json
import logging
from typing import List, Dict, Optional, Any

from openai import OpenAI, OpenAIError

from shared.pipeline.config import Config

logger = logging.getLogger(__name__)


# Use a faster/cheaper model for memory extraction if available
MEMORY_MODEL = Config.OPENAI_MODEL  # Can be overridden to use a faster model


def extract_conversation_memory(
    conversation_history: Optional[List[Dict[str, str]]] = None,
    current_question: str = ""
) -> Dict[str, Any]:
    """
    Use LLM to extract key facts and conversation state from history.

    Returns:
        {
            "facts": ["Customer has a Nexon EV", "Customer prefers portable chargers", ...],
            "answered_questions": ["vehicle type", "charger preference", "three-phase availability"],
            "do_not_ask": ["Do not ask about vehicle - customer said Nexon EV", ...],
            "summary": "Customer with Nexon EV looking for portable charger. Has confirmed three-phase power."
        }
    """
    if not conversation_history or len(conversation_history) < 2:
        return {
            "facts": [],
            "answered_questions": [],
            "do_not_ask": [],
            "summary": ""
        }

    # Build conversation text for analysis
    conversation_text = _format_conversation(conversation_history)

    try:
        memory = _extract_memory_with_llm(conversation_text, current_question)
        logger.info(f"Extracted {len(memory.get('facts', []))} facts, {len(memory.get('do_not_ask', []))} rules")
        return memory
    except Exception as e:
        logger.warning(f"LLM memory extraction failed: {e}, using fallback")
        return _extract_memory_fallback(conversation_history)


def _format_conversation(history: List[Dict[str, str]]) -> str:
    """Format conversation history for LLM analysis."""
    lines = []
    for msg in history:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_memory_with_llm(conversation_text: str, current_question: str) -> Dict[str, Any]:
    """Use LLM to extract memory from conversation."""
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    system_prompt = """You are a conversation analyzer for an EV charger sales chatbot.
Analyze the conversation and extract:
1. KEY FACTS: What has the customer told us? (vehicle, preferences, electrical setup, budget, concerns, etc.)
2. ANSWERED QUESTIONS: What questions has the customer already answered?
3. DO NOT ASK RULES: Generate specific rules about what the assistant should NOT ask again.

Output ONLY valid JSON in this exact format:
{
    "facts": ["fact 1", "fact 2", ...],
    "answered_questions": ["topic 1", "topic 2", ...],
    "do_not_ask": ["Do not ask about X - customer already said Y", ...],
    "summary": "Brief 1-2 sentence summary of customer situation"
}

IMPORTANT:
- If customer answered "yes" or "no" to any question, that's an answered question
- Include implicit information (e.g., if they confirmed three-phase, they have suitable electrical setup)
- Be specific in do_not_ask rules - explain WHY not to ask
- Output ONLY the JSON, no markdown, no explanation"""

    user_prompt = f"""CONVERSATION:
{conversation_text}

CURRENT QUESTION: {current_question if current_question else "(none yet)"}

Extract the memory from this conversation. What has the customer already told us?"""

    try:
        response = client.chat.completions.create(
            model=MEMORY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=500,
            timeout=10  # Quick timeout for memory extraction
        )

        raw_text = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                raw_text = choice.message.content.strip()

        return _parse_memory_response(raw_text)

    except OpenAIError as e:
        logger.error(f"OpenAI API error during memory extraction: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during memory extraction: {e}")
        raise


def _parse_memory_response(raw_text: str) -> Dict[str, Any]:
    """Parse LLM response into memory structure."""
    import re

    # Remove markdown code blocks if present
    text = re.sub(r'^```json?\s*', '', raw_text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    try:
        parsed = json.loads(text)

        # Validate structure
        return {
            "facts": parsed.get("facts", []) if isinstance(parsed.get("facts"), list) else [],
            "answered_questions": parsed.get("answered_questions", []) if isinstance(parsed.get("answered_questions"), list) else [],
            "do_not_ask": parsed.get("do_not_ask", []) if isinstance(parsed.get("do_not_ask"), list) else [],
            "summary": parsed.get("summary", "") if isinstance(parsed.get("summary"), str) else ""
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse memory JSON: {e}")
        # Return empty memory
        return {
            "facts": [],
            "answered_questions": [],
            "do_not_ask": [],
            "summary": ""
        }


def _extract_memory_fallback(history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Fallback memory extraction using simple heuristics.
    Used when LLM extraction fails.
    """
    facts = []
    answered = []
    do_not_ask = []

    # Combine user messages
    user_text = " ".join(
        msg.get("content", "").lower()
        for msg in history
        if msg.get("role") == "user"
    )

    # Simple pattern-based extraction
    # Vehicle detection
    vehicles = {
        "nexon": "Nexon EV", "curvv": "Curvv EV", "punch": "Punch EV",
        "tiago": "Tiago EV", "creta": "Creta EV", "ioniq": "Ioniq 5",
        "be6": "BE6", "xev": "XEV 9e", "kona": "Kona", "zs": "MG ZS EV"
    }
    for key, name in vehicles.items():
        if key in user_text:
            facts.append(f"Customer has {name}")
            answered.append("vehicle")
            do_not_ask.append(f"Do not ask about vehicle - customer has {name}")
            break

    # Preference detection
    if "portable" in user_text:
        facts.append("Customer prefers portable charger")
        answered.append("charger preference")
        do_not_ask.append("Do not ask about portable vs wall-mounted - customer wants portable")
    elif "wall" in user_text or "fixed" in user_text:
        facts.append("Customer prefers wall-mounted charger")
        answered.append("charger preference")
        do_not_ask.append("Do not ask about portable vs wall-mounted - customer wants wall-mounted")

    # Three-phase detection - check for yes/no responses after three-phase questions
    for i, msg in enumerate(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if "three-phase" in content or "three phase" in content or "3-phase" in content:
                # Check next user response
                if i + 1 < len(history):
                    next_msg = history[i + 1]
                    if next_msg.get("role") == "user":
                        response = next_msg.get("content", "").lower().strip()
                        if response in ["yes", "yeah", "yep", "yup", "sure", "correct"] or response.startswith("yes"):
                            facts.append("Customer has three-phase power")
                            answered.append("three-phase availability")
                            do_not_ask.append("Do not ask about three-phase - customer confirmed they have it")
                        elif response in ["no", "nope", "single phase", "don't", "dont"]:
                            facts.append("Customer has single-phase power only")
                            answered.append("three-phase availability")
                            do_not_ask.append("Do not ask about three-phase - customer has single-phase only")

    summary = "; ".join(facts) if facts else ""

    return {
        "facts": facts,
        "answered_questions": answered,
        "do_not_ask": do_not_ask,
        "summary": summary
    }


def build_memory_context(memory: Dict[str, Any]) -> str:
    """
    Build a context string from memory to inject into the system prompt.
    """
    if not memory or not any([memory.get("facts"), memory.get("do_not_ask"), memory.get("summary")]):
        return ""

    parts = []

    parts.append("CONVERSATION MEMORY")

    if memory.get("summary"):
        parts.append(f"Summary: {memory['summary']}")

    if memory.get("facts"):
        parts.append("\nKnown facts about this customer:")
        for fact in memory["facts"]:
            parts.append(f"- {fact}")

    if memory.get("do_not_ask"):
        parts.append("\nIMPORTANT - Questions already answered (DO NOT ASK AGAIN):")
        for rule in memory["do_not_ask"]:
            parts.append(f"- {rule}")

    return "\n".join(parts)
