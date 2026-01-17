"""
Delayed nudge generation for re-engaging customers after ~60 seconds of inactivity.

This module generates contextual follow-up messages with product videos/links
to nudge customers back into the conversation.
"""

import random
from typing import Optional, List, Dict

# Product video links for nudges
PRODUCT_VIDEOS = {
    "aveo_pro": {
        "name": "Aveo Pro",
        "demo": "https://www.youtube.com/watch?v=lLsW4FOzvkI",
        "unboxing": "https://www.youtube.com/watch?v=p9yQ7u8PtvU",
    },
    "aveo_36": {
        "name": "Aveo 3.6",
        "demo": "https://www.youtube.com/watch?v=WhImrUFSWig",
        "unboxing": "https://www.youtube.com/watch?v=BmkXONKMY_E",
    },
    "aveo_36_plus": {
        "name": "Aveo 3.6 Plus",
        "demo": "https://www.youtube.com/watch?v=P1tTSG5mRM8",
        "unboxing": "https://www.youtube.com/watch?v=BmkXONKMY_E",
    },
    "aveo_x1": {
        "name": "Aveo X1",
        "demo": "https://www.youtube.com/watch?v=ifzAiPsmGmo",
        "unboxing": "https://www.youtube.com/watch?v=IZjbINVCLM4",
    },
    "dash": {
        "name": "Dash",
        "demo": "https://www.youtube.com/watch?v=k--FvtBQbyk",
        "unboxing": "https://www.youtube.com/watch?v=4b7AKG6kQGE",
    },
    "dash_aio": {
        "name": "Dash AIO",
        "demo": "https://www.youtube.com/watch?v=sYb5JYSWSm4",
    },
    "spyder": {
        "name": "Spyder AIO",
        "demo": "https://www.youtube.com/watch?v=p_epq10rMsM",
        "unboxing": "https://www.youtube.com/watch?v=a6LKXygpBDE",
    },
    "polar": {
        "name": "Polar Series",
        "demo": "https://www.youtube.com/watch?v=g9tKl4H4BZM",
    },
    "duos": {
        "name": "Duos",
        "demo": "https://www.youtube.com/watch?v=bnjGlv2-6_E",
    },
}

# General YouTube channel link
YOUTUBE_CHANNEL = "https://www.youtube.com/@zevpoint"
PRODUCT_PLAYLIST = "https://www.youtube.com/playlist?list=PL_5jfOb-nyjIRc1nEMXJ6KsWfXietTdE4"

# Re-engagement question templates
REENGAGEMENT_QUESTIONS = [
    "Would you like me to explain anything in more detail?",
    "Do you have any questions about the installation process?",
    "Is there anything holding you back from making a decision?",
    "Would you like to know about the warranty coverage?",
    "Should I help you compare a couple of options?",
    "Do you need help understanding the electrical requirements?",
]

# Vehicle to recommended product mapping
VEHICLE_PRODUCT_MAP = {
    # 3.3 kW vehicles -> Aveo 3.6 series
    "nexon prime": "aveo_36",
    "tiago": "aveo_36",
    "tigor": "aveo_36",
    "comet": "aveo_36",
    "ec3": "aveo_36",
    # 7 kW vehicles -> Aveo Pro / Dash
    "nexon": "aveo_pro",
    "nexon ev": "aveo_pro",
    "nexon max": "aveo_pro",
    "punch": "aveo_pro",
    "curvv": "aveo_pro",
    "harrier": "aveo_pro",
    "zs": "dash",
    "zs ev": "dash",
    "windsor": "aveo_pro",
    "xuv400": "aveo_pro",
    "kona": "aveo_pro",
    "atto": "aveo_pro",
    "seal": "aveo_pro",
    # 11 kW vehicles -> Aveo X1
    "creta": "aveo_x1",
    "creta ev": "aveo_x1",
    "ioniq": "aveo_x1",
    "ioniq 5": "aveo_x1",
    "ev6": "aveo_x1",
    "be6": "aveo_x1",
    "xev 9e": "aveo_x1",
    "i4": "aveo_x1",
    "xc40": "aveo_x1",
}


def detect_product_from_conversation(conversation_history: List[Dict[str, str]]) -> Optional[str]:
    """
    Detect which product was discussed in the conversation.
    Returns product key or None.
    """
    if not conversation_history:
        return None

    # Combine all messages
    all_text = " ".join(
        msg.get("content", "").lower()
        for msg in conversation_history
    )

    # Check for specific product mentions
    product_keywords = {
        "aveo pro": "aveo_pro",
        "aveo 3.6 plus": "aveo_36_plus",
        "aveo 3.6": "aveo_36",
        "aveo x1": "aveo_x1",
        "dash aio": "dash_aio",
        "dash": "dash",
        "spyder": "spyder",
        "polar": "polar",
        "duos": "duos",
    }

    for keyword, product_key in product_keywords.items():
        if keyword in all_text:
            return product_key

    return None


def detect_vehicle_from_conversation(conversation_history: List[Dict[str, str]]) -> Optional[str]:
    """
    Detect which vehicle was mentioned in the conversation.
    Returns vehicle name or None.
    """
    if not conversation_history:
        return None

    # Combine user messages
    user_text = " ".join(
        msg.get("content", "").lower()
        for msg in conversation_history
        if msg.get("role") == "user"
    )

    # Check for vehicle mentions
    for vehicle in VEHICLE_PRODUCT_MAP.keys():
        if vehicle in user_text:
            return vehicle

    return None


def generate_delayed_nudge(
    conversation_history: Optional[List[Dict[str, str]]] = None,
    last_answer: Optional[str] = None,
    last_nudge: Optional[str] = None
) -> Optional[str]:
    """
    Generate a delayed nudge message to re-engage the customer.

    This nudge is shown ~60 seconds after the last bot message if the
    customer hasn't responded. It includes:
    - A relevant product video (if product/vehicle is known)
    - A question to push the conversation forward

    Args:
        conversation_history: Previous messages in the conversation
        last_answer: The last answer sent to the customer
        last_nudge: The immediate nudge sent with the last answer

    Returns:
        Delayed nudge message or None if not appropriate
    """
    if not conversation_history or len(conversation_history) < 2:
        return None

    # Don't send delayed nudge if last message already had a question
    if last_nudge and "?" in last_nudge:
        # Already asked a question in the immediate nudge
        # Use a different approach - share a resource
        pass

    # Detect product or vehicle from conversation
    product_key = detect_product_from_conversation(conversation_history)
    vehicle = detect_vehicle_from_conversation(conversation_history)

    # If no product but have vehicle, map vehicle to recommended product
    if not product_key and vehicle:
        product_key = VEHICLE_PRODUCT_MAP.get(vehicle)

    # Generate nudge based on context
    if product_key and product_key in PRODUCT_VIDEOS:
        product_info = PRODUCT_VIDEOS[product_key]
        video_url = product_info.get("demo") or product_info.get("unboxing")
        product_name = product_info["name"]

        # Pick a random re-engagement question
        question = random.choice(REENGAGEMENT_QUESTIONS)

        return f"Here's a quick video of the {product_name} in action: {video_url}. {question}"

    # No specific product - share general playlist
    if vehicle:
        question = random.choice(REENGAGEMENT_QUESTIONS)
        return f"You can see all our chargers in action here: {PRODUCT_PLAYLIST}. {question}"

    # Generic delayed nudge with question only
    return random.choice(REENGAGEMENT_QUESTIONS)


def generate_delayed_nudge_for_phase(
    phase: str,
    product_key: Optional[str] = None,
    vehicle: Optional[str] = None
) -> Optional[str]:
    """
    Generate delayed nudge based on conversation phase.

    Phases:
    - discovery: Customer hasn't shared vehicle yet
    - matching: Vehicle known, finding right charger
    - recommendation: Charger recommended, awaiting decision
    - closing: Product selected, moving toward purchase
    """
    if phase == "discovery":
        return "Which electric vehicle do you have? This helps me recommend the right charger for you."

    elif phase == "matching":
        if vehicle:
            product_key = VEHICLE_PRODUCT_MAP.get(vehicle.lower())
            if product_key and product_key in PRODUCT_VIDEOS:
                video = PRODUCT_VIDEOS[product_key].get("demo", PRODUCT_PLAYLIST)
                return f"Here's a video showing chargers that work great with your vehicle: {video}. Portable or wall-mounted - which would work better for you?"
        return "Do you prefer a portable charger you can take anywhere, or a wall-mounted one for permanent installation?"

    elif phase == "recommendation":
        if product_key and product_key in PRODUCT_VIDEOS:
            video = PRODUCT_VIDEOS[product_key].get("demo", PRODUCT_PLAYLIST)
            name = PRODUCT_VIDEOS[product_key]["name"]
            return f"You can see the {name} in action here: {video}. Any questions before you decide?"
        return "Any questions about the charger I recommended? Happy to clarify anything."

    elif phase == "closing":
        return "Ready to order? I can share the product link, or would you prefer to chat with our sales team?"

    return None
