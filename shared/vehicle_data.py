"""
Vehicle charging data - single source of truth for all vehicle-related logic.

This file defines:
1. Vehicle variants and their charging capacities
2. Auto-generates ambiguity detection based on the data

To add a new vehicle:
- Add it to VEHICLE_CHARGING_DATA with its kW capacity
- The system will automatically detect if it creates ambiguity with existing entries

Format: "Full Vehicle Name": kW_capacity
- Use the full official name (e.g., "Nexon EV Prime" not just "Nexon")
- The system extracts base names automatically for ambiguity detection
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# =============================================================================
# VEHICLE CHARGING DATA - Single Source of Truth
# =============================================================================
# Format: "Vehicle Name": charging_capacity_kW
#
# IMPORTANT: Use full variant names to enable proper disambiguation
# e.g., "Nexon EV Prime" and "Nexon EV Max" instead of just "Nexon"

VEHICLE_CHARGING_DATA: Dict[str, float] = {
    # 3.3 kW vehicles
    "Nexon EV Prime": 3.3,
    "Tiago EV": 3.3,
    "Tigor EV": 3.3,
    "MG Comet": 3.3,
    "Citroen eC3": 3.3,

    # 7 kW vehicles
    "Nexon EV Max": 7.0,
    "Punch EV": 7.0,
    "Curvv EV": 7.0,
    "Harrier EV": 7.0,
    "MG ZS EV": 7.0,
    "MG Windsor": 7.0,
    "XUV400": 7.0,
    "Hyundai Kona": 7.0,
    "BYD Atto 3": 7.0,
    "BYD Seal": 7.0,

    # 11 kW vehicles
    "Hyundai Creta EV": 11.0,
    "Hyundai Ioniq 5": 11.0,
    "Kia EV6": 11.0,
    "Mahindra BE6": 11.0,
    "Mahindra XEV 9e": 11.0,
    "BMW i4": 11.0,
    "Volvo XC40": 11.0,
}


# =============================================================================
# AUTO-GENERATED AMBIGUITY MAP
# =============================================================================

def _extract_base_name(full_name: str) -> str:
    """
    Extract the base vehicle name that a user might type.
    e.g., "Nexon EV Prime" -> "nexon"
         "Hyundai Creta EV" -> "creta"
         "MG ZS EV" -> "zs"
    """
    name_lower = full_name.lower()

    # Remove common prefixes (brand names)
    prefixes_to_remove = ["hyundai", "mahindra", "tata", "mg", "byd", "kia", "bmw", "volvo", "citroen"]
    for prefix in prefixes_to_remove:
        if name_lower.startswith(prefix + " "):
            name_lower = name_lower[len(prefix) + 1:]
            break

    # Remove common suffixes
    suffixes_to_remove = ["ev", "prime", "max", "plus", "long range", "standard range"]
    words = name_lower.split()
    base_words = []
    for word in words:
        if word not in suffixes_to_remove:
            base_words.append(word)

    # Return the first significant word as base name
    if base_words:
        return base_words[0].strip()

    # Fallback: return first word
    return words[0] if words else name_lower


def _extract_variant_identifier(full_name: str, base_name: str) -> str:
    """
    Extract what makes this variant unique.
    e.g., "Nexon EV Prime" with base "nexon" -> "prime"
         "Nexon EV Max" with base "nexon" -> "max"
    """
    name_lower = full_name.lower()

    # Remove base name and common words to find the distinguishing part
    words = name_lower.split()
    variant_words = []

    skip_words = {"ev", "hyundai", "mahindra", "tata", "mg", "byd", "kia", "bmw", "volvo", "citroen", base_name}

    for word in words:
        if word not in skip_words:
            variant_words.append(word)

    return " ".join(variant_words) if variant_words else full_name


def build_ambiguity_map() -> Dict[str, Dict]:
    """
    Auto-generate ambiguity map from VEHICLE_CHARGING_DATA.

    A base name is ambiguous if:
    - Multiple full vehicle names share the same base name
    - AND they have DIFFERENT charging capacities

    Returns dict with structure:
    {
        "nexon": {
            "variants": [("Nexon EV Prime", 3.3), ("Nexon EV Max", 7.0)],
            "variant_identifiers": ["prime", "max"],
            "capacities": {3.3, 7.0},
            "is_ambiguous": True
        }
    }
    """
    # Group vehicles by base name
    base_name_groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for full_name, kw in VEHICLE_CHARGING_DATA.items():
        base = _extract_base_name(full_name)
        base_name_groups[base].append((full_name, kw))

    # Build ambiguity map
    ambiguity_map: Dict[str, Dict] = {}

    for base_name, variants in base_name_groups.items():
        capacities = set(kw for _, kw in variants)

        # Extract variant identifiers for disambiguation question
        variant_identifiers = []
        for full_name, _ in variants:
            identifier = _extract_variant_identifier(full_name, base_name)
            if identifier and identifier != base_name:
                variant_identifiers.append(identifier)

        ambiguity_map[base_name] = {
            "variants": variants,
            "variant_identifiers": variant_identifiers,
            "capacities": capacities,
            "is_ambiguous": len(capacities) > 1  # Ambiguous if different kW values
        }

    # Log ambiguous vehicles
    ambiguous = [name for name, info in ambiguity_map.items() if info["is_ambiguous"]]
    if ambiguous:
        logger.info(f"Auto-detected ambiguous vehicles: {ambiguous}")

    return ambiguity_map


# Build map at module load time
AMBIGUITY_MAP = build_ambiguity_map()


def detect_ambiguous_vehicle(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Optional[Dict]:
    """
    Check if user mentioned an ambiguous vehicle name without specifying variant.

    Returns disambiguation info if ambiguous, None if clear or already specified.
    """
    query_lower = query.lower()

    # Build full conversation text to check if variant was already specified
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            history_text += " " + msg.get("content", "").lower()

    full_text = history_text + " " + query_lower

    for base_name, info in AMBIGUITY_MAP.items():
        if not info["is_ambiguous"]:
            continue

        # Check if base name is mentioned in current query
        if not re.search(rf'\b{re.escape(base_name)}\b', query_lower):
            continue

        # Check if a specific variant identifier is already mentioned
        variant_specified = False
        variant_identifiers = info["variant_identifiers"]

        for identifier in variant_identifiers:
            # Check for "base_name + identifier" pattern (e.g., "nexon prime", "nexon max")
            pattern = rf'\b{re.escape(base_name)}\s*(?:ev\s*)?{re.escape(identifier)}\b'
            if re.search(pattern, full_text):
                variant_specified = True
                break

            # Also check if identifier alone is in current query when base was in history
            if base_name in history_text and re.search(rf'\b{re.escape(identifier)}\b', query_lower):
                variant_specified = True
                break

        if not variant_specified:
            # Generate clarification question dynamically
            variant_names = [f"{id.title()}" for id in variant_identifiers if id]

            if len(variant_names) >= 2:
                question = f"Which {base_name.title()} variant do you have - {', '.join(variant_names[:-1])}, or {variant_names[-1]}?"
            else:
                question = f"Which {base_name.title()} variant do you have?"

            # Generate variant descriptions with kW
            variant_descriptions = [
                f"{full_name} ({kw}kW)"
                for full_name, kw in info["variants"]
            ]

            return {
                "vehicle": base_name,
                "variants": variant_descriptions,
                "variant_identifiers": variant_identifiers,
                "question": question,
                "capacities": list(info["capacities"])
            }

    return None


def get_vehicle_capacity(vehicle_name: str) -> Optional[float]:
    """
    Get charging capacity for a vehicle name.
    Tries exact match first, then fuzzy match.
    """
    vehicle_lower = vehicle_name.lower()

    # Exact match
    for name, kw in VEHICLE_CHARGING_DATA.items():
        if name.lower() == vehicle_lower:
            return kw

    # Fuzzy match - check if vehicle_name is contained in any full name
    for name, kw in VEHICLE_CHARGING_DATA.items():
        if vehicle_lower in name.lower() or name.lower() in vehicle_lower:
            return kw

    # Check by base name (returns None if ambiguous)
    for base_name, info in AMBIGUITY_MAP.items():
        if base_name in vehicle_lower:
            if not info["is_ambiguous"]:
                # Unambiguous - return the single capacity
                return list(info["capacities"])[0]
            # Ambiguous - can't determine
            return None

    return None


def get_recommended_charger_kw(vehicle_capacity: float) -> float:
    """
    Get recommended charger kW based on vehicle capacity.
    """
    if vehicle_capacity <= 3.3:
        return 3.6
    elif vehicle_capacity <= 7:
        return 7.5
    elif vehicle_capacity <= 11:
        return 11.0
    else:
        return 22.0
