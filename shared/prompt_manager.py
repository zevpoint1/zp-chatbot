import os
import re
import logging
from typing import List, Dict, Optional

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
# Conversation Context Extraction
# ----------------------------------------------------------------------
def extract_conversation_context(conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict:
    """Extract key information from conversation history."""
    context = {
        "vehicle": None,
        "vehicle_category": None,  # A, B, C, D
        "preference": None,  # portable, wall-mounted
        "wants_app": None,  # True, False, None (unknown)
        "product_mentioned": None,
        "has_charger": False,  # True if customer already owns a charger
        "needs_installation_only": False,  # True if customer only needs installation
        "has_three_phase": None,  # True, False, None (unknown)
    }

    if not conversation_history:
        return context

    # Combine all user messages
    user_text = ""
    for msg in conversation_history:
        if msg.get("role") == "user":
            user_text += " " + msg.get("content", "").lower()

    # Vehicle detection with categories
    # Include common typos and variations
    vehicle_map = {
        # Category A (3.3 kW)
        "nexon prime": ("Nexon Prime", "A"),
        "tiago": ("Tiago EV", "A"),
        "tigor": ("Tigor EV", "A"),
        "comet": ("MG Comet", "A"),
        "ec3": ("Citroen eC3", "A"),

        # Category B (7 kW)
        "nexon ev": ("Nexon EV", "B"),
        "nexon max": ("Nexon Max", "B"),
        "nexon": ("Nexon EV", "B"),
        "punch": ("Punch EV", "B"),
        "curvv": ("Curvv EV", "B"),
        "harrier": ("Harrier EV", "B"),
        "zs ev": ("MG ZS EV", "B"),
        "zs": ("MG ZS EV", "B"),
        "windsor": ("MG Windsor", "B"),
        "xuv400": ("XUV400", "B"),
        "xuv 400": ("XUV400", "B"),
        "kona": ("Hyundai Kona", "B"),
        "atto": ("BYD Atto 3", "B"),
        "seal": ("BYD Seal", "B"),

        # Category C (11 kW) - with common typos
        "creta ev": ("Hyundai Creta EV", "C"),
        "creta": ("Hyundai Creta EV", "C"),
        "ioniq": ("Ioniq 5", "C"),
        "ev6": ("Kia EV6", "C"),
        "be 6": ("Mahindra BE6", "C"),
        "be6": ("Mahindra BE6", "C"),
        "xev 9e": ("XEV 9e", "C"),
        "xev9e": ("XEV 9e", "C"),
        "xe9": ("XEV 9e", "C"),
        "xev9": ("XEV 9e", "C"),
        "9e": ("XEV 9e", "C"),
        "i4": ("BMW i4", "C"),
        "xc40": ("Volvo XC40", "C"),
        "model y": ("Tesla Model Y", "C"),
    }

    # Also check assistant responses for confirmed vehicles
    # This catches cases where user typed shorthand but bot confirmed full name
    assistant_text = ""
    for msg in conversation_history:
        if msg.get("role") == "assistant":
            assistant_text += " " + msg.get("content", "").lower()

    # Combined text for searching (user input + bot confirmations)
    combined_text = user_text + " " + assistant_text

    # Check for Nexon Prime specifically (before generic nexon)
    if "nexon prime" in combined_text or ("prime" in user_text and "nexon" in combined_text):
        context["vehicle"] = "Nexon Prime"
        context["vehicle_category"] = "A"
    elif "nexon max" in combined_text or ("max" in user_text and "nexon" in combined_text):
        context["vehicle"] = "Nexon Max"
        context["vehicle_category"] = "B"
    else:
        # First check user text, then check if bot confirmed a vehicle
        for keyword, (vehicle, category) in vehicle_map.items():
            if keyword in user_text:
                context["vehicle"] = vehicle
                context["vehicle_category"] = category
                break

        # If not found in user text, check bot's confirmations
        if not context["vehicle"]:
            for keyword, (vehicle, category) in vehicle_map.items():
                if keyword in assistant_text:
                    context["vehicle"] = vehicle
                    context["vehicle_category"] = category
                    break

    # Preference detection
    if "portable" in user_text:
        context["preference"] = "portable"
    elif "wall" in user_text or "fixed" in user_text or "permanent" in user_text:
        context["preference"] = "wall-mounted"

    # App preference
    if "app" in user_text or "smart" in user_text or "wifi" in user_text or "bluetooth" in user_text:
        context["wants_app"] = True
    elif "simple" in user_text or "basic" in user_text or "no app" in user_text:
        context["wants_app"] = False

    # Product mentions
    products = ["aveo pro", "aveo x1", "aveo 3.6", "dash aio", "dash", "spyder", "polar pro", "polar x1", "polar max"]
    for product in products:
        if product in user_text:
            context["product_mentioned"] = product.title()
            break

    # Detect if customer already has a charger
    has_charger_patterns = [
        "already have", "i have the charger", "have a charger", "have charger",
        "got the charger", "bought the charger", "purchased", "own a charger",
        "have 7kw", "have 3.6kw", "have 11kw", "have 22kw"
    ]
    for pattern in has_charger_patterns:
        if pattern in user_text:
            context["has_charger"] = True
            break

    # Detect if customer needs installation only
    install_only_patterns = [
        "need installation", "only installation", "just installation",
        "installation only", "install my charger", "get it installed",
        "looking for installation", "want installation"
    ]
    for pattern in install_only_patterns:
        if pattern in user_text:
            context["needs_installation_only"] = True
            break

    # Detect three-phase power confirmation
    # Check both user text and assistant confirmations in sequence
    if conversation_history:
        for i, msg in enumerate(conversation_history):
            msg_content = msg.get("content", "").lower()
            msg_role = msg.get("role", "")

            # If bot asked about three-phase
            if msg_role == "assistant" and ("three-phase" in msg_content or "three phase" in msg_content or "3-phase" in msg_content or "3 phase" in msg_content):
                # Check if user responded positively in the next message
                if i + 1 < len(conversation_history):
                    next_msg = conversation_history[i + 1]
                    if next_msg.get("role") == "user":
                        user_response = next_msg.get("content", "").lower().strip()
                        # Positive responses
                        if user_response in ["yes", "yeah", "yep", "yup", "sure", "correct", "right", "i do", "we do", "have it", "yes i do", "yes we do"] or user_response.startswith("yes"):
                            context["has_three_phase"] = True
                        # Negative responses
                        elif user_response in ["no", "nope", "don't", "dont", "single", "single phase", "no i don't", "no we don't"]:
                            context["has_three_phase"] = False

    # Also check direct mentions in user text
    if "three phase" in user_text or "three-phase" in user_text or "3 phase" in user_text or "3-phase" in user_text:
        if "have three" in user_text or "got three" in user_text or "yes" in user_text:
            context["has_three_phase"] = True
        elif "no three" in user_text or "don't have three" in user_text or "single phase" in user_text:
            context["has_three_phase"] = False

    return context


def get_conversation_phase(conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Determine conversation phase."""
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
    """
    Backwards compatible wrapper for extract_conversation_context.
    Returns dict with vehicle_mentioned, vehicle_name, installation_type, selected_product, use_case.
    """
    ctx = extract_conversation_context(conversation_history)
    return {
        "vehicle_mentioned": ctx["vehicle"] is not None,
        "vehicle_name": ctx["vehicle"],
        "installation_type": ctx["preference"],
        "selected_product": ctx["product_mentioned"],
        "use_case": None  # Not tracked in new version
    }


# ----------------------------------------------------------------------
# Build Prompt
# ----------------------------------------------------------------------
def build_prompt(
    intents: list,
    question: str = None,
    context: str = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    conversation_state: str = "ACTIVE"
) -> str:
    """Build the system prompt with conversation context."""

    # Load base prompt
    base = load_prompt_file("base")
    parts = [base]

    # Add sales prompt for sales intent
    if "sales" in intents:
        sales = load_prompt_file("sales")
        if sales:
            parts.append(sales)

    # Extract conversation context
    conv_ctx = extract_conversation_context(conversation_history)
    phase = get_conversation_phase(conversation_history)

    # Add conversation state
    state_info = f"""
CURRENT CONVERSATION STATE
Phase: {phase.upper()}
"""

    if conv_ctx["vehicle"]:
        state_info += f"Vehicle: {conv_ctx['vehicle']} (Category {conv_ctx['vehicle_category']})\n"
        state_info += "RULE: Do NOT ask for vehicle again.\n"

    if conv_ctx["preference"]:
        state_info += f"Preference: {conv_ctx['preference']}\n"
        state_info += "RULE: Do NOT ask portable/wall-mounted again.\n"

    if conv_ctx["wants_app"] is True:
        state_info += "Wants: App/smart features\n"
    elif conv_ctx["wants_app"] is False:
        state_info += "Wants: Simple, no app needed\n"

    if conv_ctx["product_mentioned"]:
        state_info += f"Product discussed: {conv_ctx['product_mentioned']}\n"

    if conv_ctx["has_charger"] or conv_ctx["needs_installation_only"]:
        state_info += "\nCUSTOMER ALREADY HAS A CHARGER - NEEDS INSTALLATION ONLY\n"
        state_info += "RULE: Do NOT recommend new chargers. Focus on installation service.\n"
        state_info += "RULE: Ask which charger they have (if not known) to provide correct installation requirements.\n"

    # Three-phase power status
    if conv_ctx["has_three_phase"] is True:
        state_info += "\nThree-phase power: CONFIRMED (customer said yes)\n"
        state_info += "RULE: Do NOT ask about three-phase again. Customer has confirmed they have it.\n"
    elif conv_ctx["has_three_phase"] is False:
        state_info += "\nThree-phase power: NO (customer has single-phase)\n"
        state_info += "RULE: Do NOT ask about three-phase again. Recommend chargers that work with single-phase.\n"

    parts.append(state_info)

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
            {"role": "user", "content": "nexon max"},
            {"role": "assistant", "content": "The Nexon Max supports 7kW charging. Would you prefer portable or wall-mounted?"},
        ]

        prompt = build_prompt(intents, question, context="[Sample context]", conversation_history=history)

        print("=" * 60)
        print(f"Question: {question}")
        print(f"Intents: {intents}")
        print(f"Phase: {get_conversation_phase(history)}")
        print(f"Context: {extract_conversation_context(history)}")
        print("=" * 60)
        print(prompt)
    else:
        print("Usage: python prompt_manager.py 'your question'")
