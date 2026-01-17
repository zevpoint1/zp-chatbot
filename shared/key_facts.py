"""
Key Facts Extraction Module - Lightweight regex-based extraction of important conversation facts.

This module extracts and maintains key facts from conversations that should persist
regardless of conversation length. Unlike full conversation history, key facts have
a fixed token cost (~50-100 tokens) and are always included in the system prompt.

Facts extracted:
- vehicle: The car model being discussed
- vehicle_variant: Specific variant (Prime, Max, etc.)
- charging_capacity: Confirmed kW capacity
- charger_preference: Product user is interested in
- living_situation: House, apartment, etc.
- electrical_setup: Socket type, phase, MCB info
- budget_range: Price expectations if mentioned
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class KeyFacts:
    """Container for key facts extracted from conversation."""
    vehicle: Optional[str] = None
    vehicle_variant: Optional[str] = None
    charging_capacity_kw: Optional[float] = None
    charger_preference: Optional[str] = None
    living_situation: Optional[str] = None
    electrical_setup: Dict[str, Any] = field(default_factory=dict)
    budget_mentioned: Optional[str] = None
    installation_interest: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and value != {}:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Serialize to JSON string for storage."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "KeyFacts":
        """Deserialize from JSON string."""
        if not json_str:
            return cls()
        try:
            data = json.loads(json_str)
            return cls(
                vehicle=data.get("vehicle"),
                vehicle_variant=data.get("vehicle_variant"),
                charging_capacity_kw=data.get("charging_capacity_kw"),
                charger_preference=data.get("charger_preference"),
                living_situation=data.get("living_situation"),
                electrical_setup=data.get("electrical_setup", {}),
                budget_mentioned=data.get("budget_mentioned"),
                installation_interest=data.get("installation_interest"),
            )
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse key_facts JSON: {json_str[:100]}")
            return cls()

    def to_prompt_string(self) -> str:
        """Format key facts for inclusion in system prompt."""
        facts = self.to_dict()
        if not facts:
            return ""

        lines = ["KNOWN FACTS ABOUT THIS CUSTOMER:"]

        if self.vehicle:
            vehicle_str = self.vehicle
            if self.vehicle_variant:
                vehicle_str += f" ({self.vehicle_variant})"
            if self.charging_capacity_kw:
                vehicle_str += f" - {self.charging_capacity_kw}kW charging"
            lines.append(f"- Vehicle: {vehicle_str}")

        if self.charger_preference:
            lines.append(f"- Interested in: {self.charger_preference}")

        if self.living_situation:
            lines.append(f"- Living situation: {self.living_situation}")

        if self.electrical_setup:
            setup_parts = []
            if self.electrical_setup.get("socket_type"):
                setup_parts.append(f"socket: {self.electrical_setup['socket_type']}")
            if self.electrical_setup.get("phase"):
                setup_parts.append(f"{self.electrical_setup['phase']}-phase")
            if self.electrical_setup.get("mcb"):
                setup_parts.append(f"MCB: {self.electrical_setup['mcb']}")
            if self.electrical_setup.get("earthing"):
                setup_parts.append(f"earthing: {self.electrical_setup['earthing']}")
            if setup_parts:
                lines.append(f"- Electrical setup: {', '.join(setup_parts)}")

        if self.budget_mentioned:
            lines.append(f"- Budget indication: {self.budget_mentioned}")

        if self.installation_interest is not None:
            lines.append(f"- Wants installation service: {'Yes' if self.installation_interest else 'No'}")

        return "\n".join(lines)

    def merge(self, other: "KeyFacts") -> "KeyFacts":
        """Merge another KeyFacts into this one. Other values take precedence if set."""
        if other.vehicle:
            self.vehicle = other.vehicle
        if other.vehicle_variant:
            self.vehicle_variant = other.vehicle_variant
        if other.charging_capacity_kw:
            self.charging_capacity_kw = other.charging_capacity_kw
        if other.charger_preference:
            self.charger_preference = other.charger_preference
        if other.living_situation:
            self.living_situation = other.living_situation
        if other.electrical_setup:
            self.electrical_setup.update(other.electrical_setup)
        if other.budget_mentioned:
            self.budget_mentioned = other.budget_mentioned
        if other.installation_interest is not None:
            self.installation_interest = other.installation_interest
        return self


# =============================================================================
# EXTRACTION PATTERNS
# =============================================================================

# Vehicle patterns - matches common EV models
VEHICLE_PATTERNS = [
    # Tata
    (r'\b(nexon)(?:\s+ev)?\s*(prime|max|long\s*range)?\b', 'Nexon'),
    (r'\b(tiago)(?:\s+ev)?\b', 'Tiago EV'),
    (r'\b(tigor)(?:\s+ev)?\b', 'Tigor EV'),
    (r'\b(punch)(?:\s+ev)?\b', 'Punch EV'),
    (r'\b(curvv)(?:\s+ev)?\b', 'Curvv EV'),
    (r'\b(harrier)(?:\s+ev)?\b', 'Harrier EV'),
    # Mahindra
    (r'\b(xuv\s*400)\b', 'XUV400'),
    (r'\b(be\s*6|be6)\b', 'Mahindra BE6'),
    (r'\b(xev\s*9e)\b', 'Mahindra XEV 9e'),
    # MG
    (r'\b(zs\s*ev)\b', 'MG ZS EV'),
    (r'\b(comet)\b', 'MG Comet'),
    (r'\b(windsor)\b', 'MG Windsor'),
    # Hyundai
    (r'\b(kona)(?:\s+ev)?\b', 'Hyundai Kona'),
    (r'\b(ioniq)\s*(\d+)?\b', 'Hyundai Ioniq'),
    (r'\b(creta)(?:\s+ev)?\b', 'Hyundai Creta EV'),
    # BYD
    (r'\b(atto)\s*(\d+)?\b', 'BYD Atto 3'),
    (r'\b(e6)\b', 'BYD E6'),
    (r'\b(seal)\b', 'BYD Seal'),
    (r'\b(emax)\s*(\d+)?\b', 'BYD eMAX 7'),
    # Kia
    (r'\b(ev6)\b', 'Kia EV6'),
    # Citroen
    (r'\b(ec3|e\s*c3)\b', 'Citroen eC3'),
]

# Variant patterns - extracts specific variant names
VARIANT_PATTERNS = [
    (r'\b(prime)\b', 'Prime'),
    (r'\b(max)\b', 'Max'),
    (r'\b(long\s*range)\b', 'Long Range'),
    (r'\b(standard\s*range)\b', 'Standard Range'),
    (r'\b(plus)\b', 'Plus'),
]

# Charger/product patterns
CHARGER_PATTERNS = [
    (r'\b(aveo)\s*(pro|x1|3\.6|plus)?\b', 'Aveo'),
    (r'\b(dash)\s*(aio)?\b', 'Dash'),
    (r'\b(spyder)\b', 'Spyder'),
    (r'\b(duos)\s*(7\.5|22)?\b', 'Duos'),
    (r'\b(polar)\s*(pro|x1|max)?\b', 'Polar'),
    (r'\b(nova)\s*(60|120|240)?\b', 'Nova'),
    (r'\b(titan)\b', 'Titan'),
    (r'\b(navigator)\b', 'Navigator'),
    (r'\b(3\.6\s*kw|3\.6kw)\b', '3.6kW charger'),
    (r'\b(7\.5\s*kw|7\.5kw)\b', '7.5kW charger'),
    (r'\b(11\s*kw|11kw)\b', '11kW charger'),
    (r'\b(22\s*kw|22kw)\b', '22kW charger'),
]

# Charging capacity patterns
CAPACITY_PATTERNS = [
    (r'\b(3\.3)\s*kw\b', 3.3),
    (r'\b(7)\s*kw\b', 7.0),
    (r'\b(7\.5)\s*kw\b', 7.5),
    (r'\b(11)\s*kw\b', 11.0),
    (r'\b(22)\s*kw\b', 22.0),
]

# Living situation patterns
LIVING_PATTERNS = [
    (r'\b(house|bungalow|villa|independent\s*house)\b', 'independent house'),
    (r'\b(apartment|flat|society|condo)\b', 'apartment'),
    (r'\b(office|commercial|workplace)\b', 'commercial'),
    (r'\b(parking|basement)\b', 'parking area'),
]

# Electrical setup patterns
SOCKET_PATTERNS = [
    (r'\b(15\s*amp|15a|15-amp)\s*(socket)?\b', '15A'),
    (r'\b(16\s*amp|16a|16-amp)\s*(socket)?\b', '16A'),
    (r'\b(32\s*amp|32a|32-amp)\s*(socket)?\b', '32A'),
    (r'\b(industrial\s*socket)\b', 'industrial'),
]

PHASE_PATTERNS = [
    (r'\b(single\s*phase|1\s*phase)\b', 'single'),
    (r'\b(three\s*phase|3\s*phase)\b', 'three'),
]

MCB_PATTERNS = [
    (r'\b(\d+)\s*amp\s*mcb\b', None),  # Captures the number
    (r'\bmcb\s*(\d+)\s*amp\b', None),
]

EARTHING_PATTERNS = [
    (r'\b(earthing|grounding)\s*(available|done|yes|installed)\b', 'available'),
    (r'\b(no\s*earthing|no\s*grounding)\b', 'not available'),
]

# Budget patterns
BUDGET_PATTERNS = [
    (r'\b(budget|spend|price\s*range).*?(\d+[,\d]*)\s*(k|thousand|lakh|lac)?\b', None),
    (r'\b(\d+[,\d]*)\s*(k|thousand|lakh|lac)?\s*(budget|max|maximum)\b', None),
    (r'\bunder\s*(\d+[,\d]*)\s*(k|thousand|lakh|lac)?\b', None),
]

# Installation interest patterns
INSTALLATION_PATTERNS = [
    (r'\b(need|want|require|interested\s*in)\s*(installation|install|setup|fitting)\b', True),
    (r'\b(installation|install)\s*(needed|required|wanted)\b', True),
    (r'\b(self\s*install|diy|myself)\b', False),
    (r'\b(no\s*installation|don\'t\s*need\s*install)\b', False),
]


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_vehicle(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract vehicle name and variant from text."""
    text_lower = text.lower()
    vehicle = None
    variant = None

    # Find vehicle
    for pattern, name in VEHICLE_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            vehicle = name
            # Check if variant is in the same match
            groups = match.groups()
            if len(groups) > 1 and groups[1]:
                variant = groups[1].strip().title()
            break

    # If no variant found in vehicle match, search separately
    if vehicle and not variant:
        for pattern, var_name in VARIANT_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                variant = var_name
                break

    return vehicle, variant


def extract_charger_preference(text: str) -> Optional[str]:
    """Extract charger/product preference from text."""
    text_lower = text.lower()

    for pattern, name in CHARGER_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) > 1 and groups[1]:
                return f"{name} {groups[1].strip()}"
            return name

    return None


def extract_charging_capacity(text: str) -> Optional[float]:
    """Extract confirmed charging capacity from text."""
    text_lower = text.lower()

    for pattern, kw in CAPACITY_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return kw

    return None


def extract_living_situation(text: str) -> Optional[str]:
    """Extract living situation from text."""
    text_lower = text.lower()

    for pattern, situation in LIVING_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return situation

    return None


def extract_electrical_setup(text: str) -> Dict[str, Any]:
    """Extract electrical setup details from text."""
    text_lower = text.lower()
    setup = {}

    # Socket type
    for pattern, socket_type in SOCKET_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            setup["socket_type"] = socket_type
            break

    # Phase
    for pattern, phase in PHASE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            setup["phase"] = phase
            break

    # MCB
    for pattern, _ in MCB_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            setup["mcb"] = f"{match.group(1)}A"
            break

    # Earthing
    for pattern, status in EARTHING_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            setup["earthing"] = status
            break

    return setup


def extract_budget(text: str) -> Optional[str]:
    """Extract budget indication from text."""
    text_lower = text.lower()

    for pattern, _ in BUDGET_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            amount = None
            unit = None
            for g in groups:
                if g and re.match(r'\d', g):
                    amount = g.replace(',', '')
                elif g and g.lower() in ('k', 'thousand', 'lakh', 'lac'):
                    unit = g.lower()

            if amount:
                if unit in ('k', 'thousand'):
                    return f"~Rs {amount},000"
                elif unit in ('lakh', 'lac'):
                    return f"~Rs {amount} lakh"
                else:
                    return f"~Rs {amount}"

    return None


def extract_installation_interest(text: str) -> Optional[bool]:
    """Extract installation interest from text."""
    text_lower = text.lower()

    for pattern, interest in INSTALLATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return interest

    return None


def extract_facts_from_message(text: str) -> KeyFacts:
    """Extract all key facts from a single message."""
    facts = KeyFacts()

    vehicle, variant = extract_vehicle(text)
    facts.vehicle = vehicle
    facts.vehicle_variant = variant
    facts.charging_capacity_kw = extract_charging_capacity(text)
    facts.charger_preference = extract_charger_preference(text)
    facts.living_situation = extract_living_situation(text)
    facts.electrical_setup = extract_electrical_setup(text)
    facts.budget_mentioned = extract_budget(text)
    facts.installation_interest = extract_installation_interest(text)

    return facts


def extract_facts_from_conversation(
    conversation_history: List[Dict[str, str]],
    existing_facts: Optional[KeyFacts] = None
) -> KeyFacts:
    """
    Extract key facts from entire conversation history.

    Later messages take precedence over earlier ones (user may correct themselves).
    """
    facts = existing_facts or KeyFacts()

    if not conversation_history:
        return facts

    # Process all messages to extract facts
    for msg in conversation_history:
        content = msg.get("content", "")
        role = msg.get("role", "")

        # Extract from both user and assistant messages
        # (assistant may confirm facts like "So you have a Nexon Prime")
        msg_facts = extract_facts_from_message(content)

        # Merge - later facts override earlier ones
        facts.merge(msg_facts)

    return facts


def update_facts_from_new_message(
    new_message: str,
    existing_facts: Optional[KeyFacts] = None
) -> KeyFacts:
    """
    Update existing facts with information from a new message.

    This is more efficient than re-processing entire history.
    """
    facts = existing_facts or KeyFacts()
    new_facts = extract_facts_from_message(new_message)
    return facts.merge(new_facts)


def extract_facts_from_qa_pair(
    bot_question: str,
    user_answer: str,
    existing_facts: Optional[KeyFacts] = None
) -> KeyFacts:
    """
    Extract facts from Q&A pairs where user gives short contextual answers.

    This handles cases like:
    - Bot: "Do you have three-phase power?" User: "yes"
    - Bot: "Do you prefer portable or wall-mounted?" User: "portable"
    - Bot: "Do you live in a house or apartment?" User: "house"

    The user's short answer only makes sense in context of the bot's question.
    """
    facts = existing_facts or KeyFacts()

    if not bot_question or not user_answer:
        return facts

    answer_lower = user_answer.lower().strip()
    question_lower = bot_question.lower()

    # Detect affirmative/negative responses
    affirmative = answer_lower in (
        "yes", "yeah", "yep", "yup", "ha", "haan", "ji", "ji haan",
        "correct", "right", "sure", "of course", "definitely", "absolutely",
        "i do", "i have", "we do", "we have", "available", "yes i do", "yes i have"
    )
    negative = answer_lower in (
        "no", "nope", "nah", "nahi", "na", "not really", "don't have",
        "i don't", "we don't", "not available", "no i don't"
    )

    # Three-phase power detection
    three_phase_keywords = ["three-phase", "three phase", "3-phase", "3 phase"]
    if any(kw in question_lower for kw in three_phase_keywords):
        if affirmative:
            facts.electrical_setup["phase"] = "three"
            logger.debug("Extracted from Q&A: three-phase = yes")
        elif negative:
            facts.electrical_setup["phase"] = "single"
            logger.debug("Extracted from Q&A: three-phase = no (single phase)")

    # Single-phase detection
    single_phase_keywords = ["single-phase", "single phase", "1-phase", "1 phase"]
    if any(kw in question_lower for kw in single_phase_keywords):
        if affirmative:
            facts.electrical_setup["phase"] = "single"
        elif negative:
            facts.electrical_setup["phase"] = "three"

    # Portable vs wall-mounted preference
    if "portable" in question_lower and "wall" in question_lower:
        # Question asks about preference between both
        if "portable" in answer_lower:
            facts.charger_preference = "portable"
            logger.debug("Extracted from Q&A: preference = portable")
        elif any(w in answer_lower for w in ["wall", "fixed", "mounted"]):
            facts.charger_preference = "wall-mounted"
            logger.debug("Extracted from Q&A: preference = wall-mounted")
    elif "portable" in question_lower:
        if affirmative:
            facts.charger_preference = "portable"
        elif negative:
            facts.charger_preference = "wall-mounted"
    elif "wall" in question_lower and ("mount" in question_lower or "fixed" in question_lower):
        if affirmative:
            facts.charger_preference = "wall-mounted"
        elif negative:
            facts.charger_preference = "portable"

    # Living situation
    if "house" in question_lower and "apartment" in question_lower:
        if any(w in answer_lower for w in ["house", "bungalow", "villa", "independent"]):
            facts.living_situation = "independent house"
            logger.debug("Extracted from Q&A: living = house")
        elif any(w in answer_lower for w in ["apartment", "flat", "society", "condo"]):
            facts.living_situation = "apartment"
            logger.debug("Extracted from Q&A: living = apartment")
    elif "house" in question_lower or "independent" in question_lower:
        if affirmative:
            facts.living_situation = "independent house"
    elif "apartment" in question_lower or "flat" in question_lower or "society" in question_lower:
        if affirmative:
            facts.living_situation = "apartment"

    # Earthing/grounding
    earthing_keywords = ["earthing", "grounding", "earth connection", "ground connection"]
    if any(kw in question_lower for kw in earthing_keywords):
        if affirmative:
            facts.electrical_setup["earthing"] = "available"
            logger.debug("Extracted from Q&A: earthing = available")
        elif negative:
            facts.electrical_setup["earthing"] = "not available"

    # Installation interest
    installation_keywords = ["installation", "install", "set up", "setup", "fitting"]
    if any(kw in question_lower for kw in installation_keywords):
        if "need" in question_lower or "want" in question_lower or "require" in question_lower:
            if affirmative:
                facts.installation_interest = True
                logger.debug("Extracted from Q&A: installation_interest = True")
            elif negative:
                facts.installation_interest = False

    # App preference
    if "app" in question_lower:
        if affirmative:
            # Could track app preference if we add that field
            logger.debug("Customer wants app control")
        elif negative:
            logger.debug("Customer doesn't need app control")

    return facts
