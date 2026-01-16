import os
import re
import logging
from typing import List, Dict, Optional

from shared.conversation_state import (
    update_conversation_state,
    get_followup_message,
    should_send_followup
)

logger = logging.getLogger(__name__)

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# ----------------------------------------------------------------------
# Utility: Safe file loader
# ----------------------------------------------------------------------
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
# Multi-intent detection system
# ----------------------------------------------------------------------
INTENT_PATTERNS = {
    "sales": [
        r"\bcharger\b", r"\bev charger\b", r"\bbuy\b", r"\bpurchase\b",
        r"\bneed\b", r"\blooking for\b", r"\bwant\b", r"\bget\b",
        r"\brecommend\b", r"\bsuggestion\b", r"\bprice\b", r"\bcost\b",
        r"\bvehicle\b", r"\bcar\b", r"\bev\b", r"\belectric vehicle\b",
        r"\bhome charging\b", r"\bportable\b", r"\bwall.?mount(ed)?\b",
        r"\bwarranty\b", r"\binstall(ation)?\b", r"\bshipping\b",
        r"\b(specs?|specifications?)\b"
    ],
    "agent_handoff": [
        r"\bagent\b", r"\bhuman\b", r"\bperson\b", r"\btalk to someone\b",
        r"\bcustomer (care|service|support)\b", r"\bsupport team\b", 
        r"\bcall me\b", r"\bspeak (to|with)\b"
    ],
    "metrics": [
        r"\bmetrics?\b", r"\bkpi\b", r"\bperformance\b", r"\bmeasure\b",
        r"\btracking\b", r"\bscore\b"
    ],
    "hr": [
        r"\bhr\b", r"\bhuman resources\b", r"\bpolicy\b",
        r"\battendance\b", r"\bleave\b"
    ],
    "service": [
        r"\bservice\b", r"\bsupport\b", r"\bworkflow\b", r"\bsop\b",
        r"\bmaintenance\b", r"\btroubleshoot\b", r"\brepair\b"
    ],
}


def detect_intent(text: str) -> list:
    """
    Detect user intent from their message.
    Returns list of intent strings, prioritizing agent_handoff if detected.
    """
    text = text.lower()
    matched = []
    
    # Check for handoff first - it overrides other intents
    for p in INTENT_PATTERNS["agent_handoff"]:
        if re.search(p, text):
            return ["agent_handoff"]

    # Check for sales intent
    for p in INTENT_PATTERNS["sales"]:
        if re.search(p, text):
            matched.append("sales")
            break
    
    # Check other intents if no sales detected
    if not matched:
        for intent_name, patterns in INTENT_PATTERNS.items():
            if intent_name in ["agent_handoff", "sales"]:
                continue
            for p in patterns:
                if re.search(p, text):
                    matched.append(intent_name)
                    break
    
    # Default to general if nothing matched
    if not matched:
        matched.append("general")
    
    return matched


# ----------------------------------------------------------------------
# Conversation state tracking
# ----------------------------------------------------------------------
def extract_vehicle_info(conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, any]:
    """
    Extract vehicle and preference information from conversation history.
    
    Returns dict with:
    - vehicle_mentioned: bool
    - vehicle_name: str or None
    - installation_type: str or None (portable/wall-mounted)
    - selected_product: str or None
    """
    info = {
        "vehicle_mentioned": False,
        "vehicle_name": None,
        "installation_type": None,
        "selected_product": None,
        "use_case": None  # home, office, both
    }
    
    if not conversation_history:
        return info
    
    # Vehicle brands/models to look for
    vehicle_keywords = {
        "tata": ["nexon", "tiago", "tigor", "curvv", "punch"],
        "mahindra": ["xuv400", "e20", "everito"],
        "mg": ["zs ev", "comet", "windsor"],
        "hyundai": ["kona", "ioniq"],
        "citroen": ["ec3"],
        "byd": ["e6", "atto"],
    }
    
    installation_keywords = ["portable", "wall-mounted", "wall mounted", "fixed"]
    use_case_keywords = ["home", "office", "both"]
    product_keywords = ["dash", "aveo", "spyder", "flash"]
    
    # Search through conversation history
    history_text = ""
    for msg in conversation_history:
        if msg.get("role") == "user":
            history_text += " " + msg.get("content", "").lower()
    
    # Check for vehicle mentions
    for brand, models in vehicle_keywords.items():
        if brand in history_text:
            info["vehicle_mentioned"] = True
            info["vehicle_name"] = brand.capitalize()
            
            # Check for specific model
            for model in models:
                if model in history_text:
                    info["vehicle_name"] = f"{brand.capitalize()} {model.capitalize()}"
                    break
            break
    
    # Check for installation type
    for keyword in installation_keywords:
        if keyword in history_text:
            if "portable" in keyword:
                info["installation_type"] = "portable"
            else:
                info["installation_type"] = "wall-mounted"
            break
    
    # Check for use case
    for keyword in use_case_keywords:
        if keyword in history_text:
            info["use_case"] = keyword
            break
    
    # Check for product selection
    for product in product_keywords:
        if product in history_text:
            info["selected_product"] = product.capitalize()
            break
    
    return info


def needs_qualification(question: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> bool:
    """
    Determine if question needs vehicle qualification.
    Returns True only if question is vague AND vehicle not mentioned in history.
    
    Args:
        question: User's current question
        conversation_history: List of previous messages
    
    Returns:
        bool: True if needs qualification, False otherwise
    """
    if not question:
        return False

    # Check conversation history for vehicle info
    vehicle_info = extract_vehicle_info(conversation_history)
    
    if vehicle_info["vehicle_mentioned"]:
        logger.info(f"Vehicle already mentioned: {vehicle_info['vehicle_name']} - skipping qualification")
        return False
    
    question_lower = question.lower().strip()
    
    # Vague patterns that need vehicle info
    vague_patterns = [
        r"^\s*(hi|hello|hey|hii|helo)\b",
        r"^\s*(hi|hello|hey)\s*$",
        r"\bneed.*(charger|ev charger)\b",
        r"\blooking for.*(charger|ev charger)\b",
        r"\bwant.*(charger|ev charger)\b",
        r"\bget.*(charger|ev charger)\b",
        r"\bcharger.*recommendations?\b",
        r"^\s*charger\s*$",
        r"^\s*ev charger\s*$",
    ]
    
    # Specific indicators - no qualification needed
    specific_indicators = [
        r"\b(tata|mahindra|mg|hyundai|citroen|byd|ather|ola)\b",  # brands
        r"\b(nexon|curvv|windsor|comet|kona|xuv|zs)\b",  # models
        r"\d+\s*kw",  # power rating
        r"\b(home|office|portable|wall.?mount)\b",  # installation
        r"\b(price|cost|warranty|shipping|install)\b",  # specific queries
        r"\b(dash|aveo|spyder|flash)\b",  # product names
    ]
    
    is_vague = any(re.search(pattern, question_lower) for pattern in vague_patterns)
    has_specifics = any(re.search(pattern, question_lower) for pattern in specific_indicators)
    
    result = is_vague and not has_specifics
    
    if result:
        logger.info(f"Query needs qualification: '{question}'")
    else:
        logger.info(f"Query is specific enough: '{question}'")
    
    return result


def get_conversation_phase(conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Determine which phase of the sales conversation we're in.
    
    Returns:
        str: "discovery", "matching", "specification", "closing", or "general"
    """
    if not conversation_history:
        return "discovery"
    
    vehicle_info = extract_vehicle_info(conversation_history)
    
    # Phase 1: Discovery - no vehicle mentioned
    if not vehicle_info["vehicle_mentioned"]:
        return "discovery"
    
    # Phase 5: Closing - product selected
    if vehicle_info["selected_product"]:
        return "closing"
    
    # Phase 3-4: Specification - installation type discussed
    if vehicle_info["installation_type"]:
        return "specification"
    
    # Phase 2: Matching - vehicle mentioned, need to recommend
    return "matching"


# ----------------------------------------------------------------------
# Build comprehensive prompt with state awareness
# ----------------------------------------------------------------------
def build_prompt(
    intents: list,
    question: str = None,
    context: str = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    conversation_state: str = "ACTIVE"
) -> str:
    """
    Build complete system prompt with strict state + phase control.
    """

    base = load_prompt_file("base")
    final_parts = [base]

    # ------------------------------------------------------------
    # 1. HARD STOP STATES (NO SALES, NO PHASES)
    # ------------------------------------------------------------
    if conversation_state == "HOLD":
        final_parts.append(
            """
CONVERSATION STATE: HOLD

The customer has asked to pause and think.
Rules:
- Acknowledge politely
- Do NOT ask questions
- Do NOT recommend products
- Do NOT advance the conversation
"""
        )
        return "\n\n".join(final_parts)

    if conversation_state in ["FOLLOW_UP_1", "FOLLOW_UP_2", "CLOSED"]:
        # Follow-ups and closed states must NOT use LLM-driven sales logic
        return "\n\n".join(final_parts)

    # From here onward, state is ACTIVE only
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # 2. PHASE DETECTION (SALES PROGRESSION)
    # ------------------------------------------------------------
    phase = get_conversation_phase(conversation_history)
    vehicle_info = extract_vehicle_info(conversation_history)
    is_vague = needs_qualification(question, conversation_history) if question else False

    logger.info(
        f"Prompt build | State: {conversation_state} | Phase: {phase} | Vehicle: {vehicle_info['vehicle_name']}"
    )

    # ------------------------------------------------------------
    # 3. PHASE INSTRUCTIONS (ONLY WHEN ACTIVE)
    # ------------------------------------------------------------
    if "sales" in intents:
        state_marker = f"\nCONVERSATION PHASE: {phase.upper()}\n\n"

        # Persist known context
        if vehicle_info["vehicle_mentioned"]:
            state_marker += "KNOWN CUSTOMER CONTEXT:\n"
            state_marker += f"- Vehicle: {vehicle_info['vehicle_name']}\n"

            if vehicle_info["installation_type"]:
                state_marker += f"- Installation: {vehicle_info['installation_type']}\n"

            if vehicle_info["use_case"]:
                state_marker += f"- Use Case: {vehicle_info['use_case']}\n"

            if vehicle_info["selected_product"]:
                state_marker += f"- Selected Product: {vehicle_info['selected_product']}\n"

            state_marker += (
                "\nCRITICAL RULE: Never ask for vehicle details again.\n\n"
            )

        # Phase-specific behavior
        if phase == "discovery":
            state_marker += (
                "TASK:\n"
                "Ask exactly ONE question to identify the EV model.\n"
                "Do not mention products, pricing, or specs.\n"
            )

        elif phase == "matching":
            state_marker += (
                "TASK:\n"
                f"The customer drives {vehicle_info['vehicle_name']}.\n"
                "Use context to determine max AC charging capacity.\n"
                "Recommend the correct charger category.\n"
                "Ask ONE clarifying question only if required.\n"
            )

        elif phase == "specification":
            state_marker += (
                "TASK:\n"
                "Recommend the most suitable product from context.\n"
                "Explain differences briefly if alternatives exist.\n"
                "Only mention prices if customer asks.\n"
            )

        elif phase == "closing":
            state_marker += (
                "TASK:\n"
                f"Confirm the selected product: {vehicle_info['selected_product']}.\n"
                "Answer final questions (shipping, COD, warranty) briefly.\n"
                "Offer next step without pressure.\n"
            )

        final_parts.append(state_marker)

    # ------------------------------------------------------------
    # 4. DOMAIN PROMPTS
    # ------------------------------------------------------------
    for intent in intents:
        if intent == "general":
            continue
        domain_prompt = load_prompt_file(intent)
        if domain_prompt:
            final_parts.append(domain_prompt)

    # ------------------------------------------------------------
    # 5. RAG CONTEXT WITH EXPLICIT GROUNDING INSTRUCTIONS
    # ------------------------------------------------------------
    if context and context.strip():
        final_parts.append(
            f"""
CRITICAL GROUNDING REQUIREMENT:

Below are the SPECIFIC product documents retrieved for this query. You MUST:
1. READ all documents in <relevant_product_information> tags carefully
2. EXTRACT exact product names, prices, and specifications from these documents
3. CITE specific products by name (e.g., "Zevpoint Dash", "Aveo Pro") when recommending
4. QUOTE exact prices when available (e.g., "Rs. 20,999", "Rs. 22,999")
5. NEVER generate generic responses without specific product references
6. If multiple products match, present 2-3 options with prices and key differences

DO NOT use general knowledge about EV chargers. Use ONLY the information in these documents:

<relevant_product_information>
{context}
</relevant_product_information>

GROUND YOUR RESPONSE: Reference specific products from above with their exact prices and features.
"""
        )

    # ------------------------------------------------------------
    # 6. FINAL OUTPUT CONSTRAINTS
    # ------------------------------------------------------------
    final_parts.append(
        """
RESPONSE RULES:
- Maximum 1-2 sentences per response
- One question per turn (if needed)
- No praise, no filler
- Only mention prices when customer asks
- Plain text only (no markdown: *, **, #, ##)
"""
    )

    formatting = load_prompt_file("formatting")
    if formatting:
        final_parts.append(formatting)

    return "\n\n".join([p for p in final_parts if p.strip()])

# ----------------------------------------------------------------------
# CLI Preview Mode
# ----------------------------------------------------------------------
def preview_prompt(question: str, simulate_history: bool = False):
    """
    Preview the prompt that would be generated for a question.
    Optionally simulate conversation history.
    """
    intents = detect_intent(question)
    
    # Simulate conversation history if requested
    conversation_history = None
    if simulate_history:
        conversation_history = [
            {"role": "user", "content": "Hi, I need a charger"},
            {"role": "assistant", "content": "Welcome to Zevpoint! Which EV do you drive?"},
            {"role": "user", "content": "Tata Curvv"},
        ]
    
    sample_context = "[Context would be retrieved from knowledge base here]"
    prompt = build_prompt(
        intents, 
        question, 
        context=sample_context,
        conversation_history=conversation_history
    )

    print("\n==================== INTENT(S) DETECTED ====================")
    print(", ".join(intents))
    
    if conversation_history:
        print("\n==================== SIMULATED HISTORY =====================")
        for msg in conversation_history:
            print(f"{msg['role'].upper()}: {msg['content']}")
    
    print("\n==================== CONVERSATION STATE ====================")
    phase = get_conversation_phase(conversation_history)
    vehicle_info = extract_vehicle_info(conversation_history)
    print(f"Phase: {phase}")
    print(f"Vehicle Info: {vehicle_info}")

    print("\n==================== FINAL PROMPT ==========================")
    print(prompt)
    print("============================================================\n")


if __name__ == "__main__":
    import sys
    
    # Example usage:
    # python prompt_manager.py "what about warranty?" --history
    
    simulate_history = "--history" in sys.argv
    
    if len(sys.argv) > 1:
        args = [arg for arg in sys.argv[1:] if arg != "--history"]
        if args:
            q = " ".join(args)
            preview_prompt(q, simulate_history=simulate_history)
        else:
            print("Usage: python prompt_manager.py \"your question\" [--history]")
            print("  --history: Simulate a conversation with vehicle already mentioned")
    else:
        print("Usage: python prompt_manager.py \"your question\" [--history]")
        print("\nExamples:")
        print('  python prompt_manager.py "hi"')
        print('  python prompt_manager.py "what about warranty?" --history')