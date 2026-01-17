import os
import re
import logging
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.key_facts import KeyFacts

logger = logging.getLogger(__name__)

# Use parent directory for prompts (shared is inside the main folder)
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")


def load_prompt_file(name: str) -> str:
    """Load a prompt file from the prompts directory."""
    path = os.path.join(PROMPT_DIR, f"{name}.txt")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading prompt '{name}': {e}")
    return ""


# ----------------------------------------------------------------------
# Intent Detection (Simplified)
# ----------------------------------------------------------------------
INTENT_PATTERNS = {
    "sales": [
        r"\bcharger\b", r"\bev\b", r"\bbuy\b", r"\bpurchase\b",
        r"\bneed\b", r"\blooking for\b", r"\bwant\b",
        r"\brecommend\b", r"\bprice\b", r"\bcost\b",
        r"\bvehicle\b", r"\bcar\b", r"\bportable\b", r"\bwall\b",
        r"\bwarranty\b", r"\binstall\b", r"\bshipping\b",
        r"\b(nexon|curvv|tiago|punch|creta|ioniq|xuv|zs|windsor|comet|kona)\b",
        r"\b(aveo|dash|spyder|polar|duos)\b"
    ],
    "agent_handoff": [
        r"\bagent\b", r"\bhuman\b", r"\btalk to someone\b",
        r"\bcustomer (care|service|support)\b", r"\bcall me\b"
    ],
    "service": [
        r"\brepair\b", r"\btroubleshoot\b", r"\bnot working\b",
        r"\bbroken\b", r"\bfault\b", r"\berror\b"
    ],
}


def detect_intent(text: str) -> list:
    """Detect user intent. Returns list of intents."""
    text = text.lower()

    # Agent handoff overrides everything
    for p in INTENT_PATTERNS["agent_handoff"]:
        if re.search(p, text):
            return ["agent_handoff"]

    # Check for service issues
    for p in INTENT_PATTERNS["service"]:
        if re.search(p, text):
            return ["service"]

    # Default to sales for most queries
    for p in INTENT_PATTERNS["sales"]:
        if re.search(p, text):
            return ["sales"]

    # If nothing matched, assume sales (most common)
    return ["sales"]


# ----------------------------------------------------------------------
# Dynamic Conversation Memory (LLM-based)
# ----------------------------------------------------------------------
def get_conversation_memory(
    conversation_history: Optional[List[Dict[str, str]]] = None,
    current_question: str = ""
) -> Dict:
    """
    Get dynamic conversation memory using LLM extraction.
    Falls back to simple heuristics if LLM fails.
    """
    try:
        from shared.conversation_memory import extract_conversation_memory
        return extract_conversation_memory(conversation_history, current_question)
    except Exception as e:
        logger.warning(f"Failed to import conversation_memory: {e}")
        return {
            "facts": [],
            "answered_questions": [],
            "do_not_ask": [],
            "summary": ""
        }


# ----------------------------------------------------------------------
# Legacy Context Extraction (kept for backward compatibility)
# ----------------------------------------------------------------------
def extract_conversation_context(conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
    """
    Legacy function - extracts basic context using pattern matching.
    Kept for backward compatibility with existing code.
    The new build_prompt uses get_conversation_memory() instead.
    """
    context = {
        "vehicle": None,
        "vehicle_category": None,
        "preference": None,
        "wants_app": None,
        "product_mentioned": None,
        "has_charger": False,
        "needs_installation_only": False,
        "has_three_phase": None,
    }

    if not conversation_history:
        return context

    # Combine all user messages
    user_text = ""
    for msg in conversation_history:
        if msg.get("role") == "user":
            user_text += " " + msg.get("content", "").lower()

    # Quick vehicle detection for phase determination
    vehicle_map = {
        "nexon prime": ("Nexon Prime", "A"),
        "tiago": ("Tiago EV", "A"),
        "nexon": ("Nexon EV", "B"),
        "punch": ("Punch EV", "B"),
        "curvv": ("Curvv EV", "B"),
        "creta": ("Hyundai Creta EV", "C"),
        "ioniq": ("Ioniq 5", "C"),
        "be6": ("Mahindra BE6", "C"),
    }

    for keyword, (vehicle, category) in vehicle_map.items():
        if keyword in user_text:
            context["vehicle"] = vehicle
            context["vehicle_category"] = category
            break

    # Preference detection
    if "portable" in user_text:
        context["preference"] = "portable"
    elif "wall" in user_text or "fixed" in user_text:
        context["preference"] = "wall-mounted"

    # Product mentions
    products = ["aveo pro", "aveo x1", "dash", "spyder", "polar pro", "polar x1"]
    for product in products:
        if product in user_text:
            context["product_mentioned"] = product.title()
            break

    return context


def get_conversation_phase(conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Determine conversation phase for logging."""
    ctx = extract_conversation_context(conversation_history)

    if not ctx["vehicle"]:
        return "discovery"
    if not ctx["preference"]:
        return "matching"
    if ctx["product_mentioned"]:
        return "closing"
    return "recommendation"


# Backwards compatibility alias
def extract_vehicle_info(conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
    """Backwards compatible wrapper."""
    ctx = extract_conversation_context(conversation_history)
    return {
        "vehicle_mentioned": ctx["vehicle"] is not None,
        "vehicle_name": ctx["vehicle"],
        "installation_type": ctx["preference"],
        "selected_product": ctx["product_mentioned"],
        "use_case": None
    }


# ----------------------------------------------------------------------
# Build Prompt (Updated to use dynamic memory)
# ----------------------------------------------------------------------
def build_prompt(
    intents: list,
    question: str = None,
    context: str = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    conversation_state: str = "ACTIVE",
    key_facts: "KeyFacts" = None
) -> str:
    """Build the system prompt with dynamic conversation memory and key facts."""

    # Load base prompt
    base = load_prompt_file("base")
    parts = [base]

    # Add sales prompt for sales intent
    if "sales" in intents:
        sales = load_prompt_file("sales")
        if sales:
            parts.append(sales)

    # Add key facts section (always included - fixed token cost)
    if key_facts:
        key_facts_str = key_facts.to_prompt_string()
        if key_facts_str:
            parts.append(key_facts_str)
            logger.info(f"Injected key_facts into prompt: {key_facts.to_dict()}")

    # Get dynamic conversation memory (LLM-based)
    memory = get_conversation_memory(conversation_history, question or "")

    # Build memory context section
    if memory and any([memory.get("facts"), memory.get("do_not_ask"), memory.get("summary")]):
        memory_section = []
        memory_section.append("CONVERSATION MEMORY")
        memory_section.append("The following information has been gathered from this conversation.")
        memory_section.append("Use this to maintain context and avoid asking questions already answered.")

        if memory.get("summary"):
            memory_section.append(f"\nSummary: {memory['summary']}")

        if memory.get("facts"):
            memory_section.append("\nKnown facts about this customer:")
            for fact in memory["facts"]:
                memory_section.append(f"  - {fact}")

        if memory.get("do_not_ask"):
            memory_section.append("\nCRITICAL - DO NOT ASK THESE QUESTIONS AGAIN:")
            for rule in memory["do_not_ask"]:
                memory_section.append(f"  - {rule}")

        if memory.get("answered_questions"):
            memory_section.append(f"\nTopics already discussed: {', '.join(memory['answered_questions'])}")

        parts.append("\n".join(memory_section))

    # Add conversation phase for context
    phase = get_conversation_phase(conversation_history)
    parts.append(f"\nCONVERSATION PHASE: {phase.upper()}")

    # Add RAG context if provided
    if context and context.strip():
        parts.append(f"""
RETRIEVED PRODUCT INFORMATION
Use this information to answer the customer's question. Cite specific products and prices from here.

<product_info>
{context}
</product_info>

IMPORTANT: Base your answer on the product info above. If the info doesn't answer the question, offer to connect with a specialist.
""")

    # Add formatting guidelines
    formatting = load_prompt_file("formatting")
    if formatting:
        parts.append(formatting)

    # Final reminder
    parts.append("""
FINAL REMINDER
- MAX 2 sentences
- ONE question at a time
- No prices unless asked
- No technical details unless asked
- No future-proofing unless asked
- Listen to what customer actually asks
- NEVER repeat a question the customer has already answered
""")

    return "\n\n".join([p for p in parts if p.strip()])


# ----------------------------------------------------------------------
# CLI for testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        intents = detect_intent(question)

        # Simulate some history
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Welcome to Zevpoint! Which EV do you drive?"},
            {"role": "user", "content": "BE6"},
            {"role": "assistant", "content": "Great choice! The BE6 charges at 11kW. Do you prefer portable or wall-mounted?"},
            {"role": "user", "content": "portable"},
            {"role": "assistant", "content": "The Aveo X1 is our 11kW portable charger. Do you have three-phase power?"},
            {"role": "user", "content": "yes"},
        ]

        print("=" * 60)
        print("Testing Dynamic Conversation Memory")
        print("=" * 60)

        # Test memory extraction
        memory = get_conversation_memory(history, question)
        print("\nExtracted Memory:")
        print(f"  Facts: {memory.get('facts', [])}")
        print(f"  Answered: {memory.get('answered_questions', [])}")
        print(f"  Do Not Ask: {memory.get('do_not_ask', [])}")
        print(f"  Summary: {memory.get('summary', '')}")

        print("\n" + "=" * 60)
        print("Full Prompt:")
        print("=" * 60)
        prompt = build_prompt(intents, question, context="[Sample context]", conversation_history=history)
        print(prompt)
    else:
        print("Usage: python prompt_manager.py 'your question'")
