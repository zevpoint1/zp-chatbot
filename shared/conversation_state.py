import re
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional

# Patterns indicating user wants to pause/think
PAUSE_PATTERNS = r"\b(let me think|later|decide later|not now|will check|think about|get back|call you|contact later)\b"

# Patterns indicating user is done/not interested
DONE_PATTERNS = r"\b(not interested|no thanks|don't need|don't want|never mind|cancel)\b"

# Patterns indicating user wants to continue
CONTINUE_PATTERNS = r"\b(yes|ok|sure|go ahead|interested|tell me|show me|continue)\b"


def update_conversation_state(
    current_state: str = "ACTIVE",
    user_message: Optional[str] = None,
    last_user_timestamp: Optional[datetime] = None,
    followup_count: int = 0,
    now: Optional[datetime] = None
) -> Tuple[str, int, Optional[datetime]]:
    """
    Update conversation state based on user activity and time.
    
    Args:
        current_state: Current state (ACTIVE, HOLD, FOLLOW_UP_1, FOLLOW_UP_2, CLOSED)
        user_message: Latest message from user (None if checking time-based state)
        last_user_timestamp: When user last sent a message
        followup_count: Number of follow-ups already sent (0, 1, or 2)
        now: Current time (defaults to datetime.utcnow())
    
    Returns:
        Tuple of (new_state, new_followup_count, last_user_timestamp)
    
    States:
        - ACTIVE: User is actively engaged
        - HOLD: User explicitly asked to pause ("let me think")
        - FOLLOW_UP_1: First follow-up sent after 2 hours of inactivity
        - FOLLOW_UP_2: Second follow-up sent after 12 hours of inactivity
        - CLOSED: User explicitly declined or too much time passed
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # If user sends a message, check what they said
    if user_message:
        user_message_lower = user_message.lower().strip()
        
        # Check for explicit "done/not interested" patterns
        if re.search(DONE_PATTERNS, user_message_lower):
            return "CLOSED", followup_count, now
        
        # Check for explicit "pause/think" patterns
        if re.search(PAUSE_PATTERNS, user_message_lower):
            return "HOLD", followup_count, now
        
        # Check for explicit continuation (overrides HOLD if they change their mind)
        if re.search(CONTINUE_PATTERNS, user_message_lower) and current_state == "HOLD":
            return "ACTIVE", 0, now
        
        # Any other user reply → reset to ACTIVE and reset follow-ups
        return "ACTIVE", 0, now

    # No user message → check time-based transitions
    if last_user_timestamp:
        time_elapsed = now - last_user_timestamp
        
        # If in HOLD state, don't send automatic follow-ups
        if current_state == "HOLD":
            # After 48 hours in HOLD, close conversation
            if time_elapsed > timedelta(hours=48):
                return "CLOSED", followup_count, last_user_timestamp
            return current_state, followup_count, last_user_timestamp
        
        # If already closed, stay closed
        if current_state == "CLOSED":
            return current_state, followup_count, last_user_timestamp
        
        # Automatic follow-up progression
        # Close conversation 24 hours after the second follow-up
        if time_elapsed > timedelta(hours=36) and followup_count >= 2:
            return "CLOSED", followup_count, last_user_timestamp

        
        if time_elapsed > timedelta(hours=12) and followup_count == 1:
            # Send second follow-up after 12 hours
            return "FOLLOW_UP_2", 2, last_user_timestamp
        
        if time_elapsed > timedelta(hours=2) and followup_count == 0:
            # Send first follow-up after 2 hours
            return "FOLLOW_UP_1", 1, last_user_timestamp

    # No changes needed
    return current_state, followup_count, last_user_timestamp


def get_followup_message(state: str, vehicle_name: Optional[str] = None) -> Optional[str]:
    """
    Get appropriate follow-up message based on conversation state.
    
    Args:
        state: Current conversation state
        vehicle_name: Customer's vehicle (if known)
    
    Returns:
        Follow-up message string or None
    """
    if state == "FOLLOW_UP_1":
        if vehicle_name:
            return f"Hi! Just checking in - did you have any questions about charging solutions for your {vehicle_name}?"
        return "Hi! Just checking in - do you still need help finding an EV charger?"
    
    elif state == "FOLLOW_UP_2":
        if vehicle_name:
            return f"Hi again! We're here if you'd like to discuss charger options for your {vehicle_name}. Would now be a good time?"
        return "Hi again! We're here if you need any help with EV charging. Just let us know!"
    
    elif state == "HOLD":
        # Don't send automatic messages in HOLD state
        return None
    
    return None


def should_send_followup(
    state: str,
    last_bot_message: Optional[str] = None,
    time_since_last_bot: Optional[timedelta] = None
) -> bool:
    """
    Determine if bot should send a follow-up message.
    
    Args:
        state: Current conversation state
        last_bot_message: Last message bot sent
        time_since_last_bot: Time elapsed since last bot message
    
    Returns:
        bool: True if should send follow-up, False otherwise
    """
    # Never send follow-up in these states
    if state in ["ACTIVE", "CLOSED", "HOLD"]:
        return False
    
    # Idempotency guard: prevent duplicate follow-ups
    if state == "FOLLOW_UP_1" and time_since_last_bot and time_since_last_bot < timedelta(hours=2):
        return False

    if state == "FOLLOW_UP_2" and time_since_last_bot and time_since_last_bot < timedelta(hours=12):
        return False
    
    # Don't send follow-up if we just sent a message recently
    if time_since_last_bot and time_since_last_bot < timedelta(hours=1):
        return False
    
    # Don't send follow-up if last bot message was already a follow-up
    if last_bot_message:
        followup_indicators = ["checking in", "just checking", "still need help", "would now be a good time"]
        if any(indicator in last_bot_message.lower() for indicator in followup_indicators):
            return False
    
    # Send follow-up if in FOLLOW_UP state and conditions are met
    return state in ["FOLLOW_UP_1", "FOLLOW_UP_2"]


# State transition map for documentation/debugging
STATE_TRANSITIONS = {
    "ACTIVE": {
        "user_replies": "ACTIVE (reset follow-ups)",
        "2_hours_silence": "FOLLOW_UP_1",
        "user_says_pause": "HOLD",
        "user_says_done": "CLOSED"
    },
    "FOLLOW_UP_1": {
        "user_replies": "ACTIVE (reset follow-ups)",
        "12_hours_silence": "FOLLOW_UP_2",
        "user_says_pause": "HOLD",
        "user_says_done": "CLOSED"
    },
    "FOLLOW_UP_2": {
        "user_replies": "ACTIVE (reset follow-ups)",
        "24_hours_silence": "CLOSED",
        "user_says_pause": "HOLD",
        "user_says_done": "CLOSED"
    },
    "HOLD": {
        "user_says_continue": "ACTIVE (reset follow-ups)",
        "48_hours_silence": "CLOSED",
        "user_says_done": "CLOSED"
    },
    "CLOSED": {
        "stays": "CLOSED (terminal state)"
    }
}


def get_state_description(state: str) -> str:
    """Get human-readable description of a state."""
    descriptions = {
        "ACTIVE": "User is actively engaged in conversation",
        "HOLD": "User requested to pause/think - no automatic follow-ups",
        "FOLLOW_UP_1": "First follow-up sent after 2 hours of inactivity",
        "FOLLOW_UP_2": "Second follow-up sent after 12 hours of inactivity",
        "CLOSED": "Conversation ended (user declined or too much time elapsed)"
    }
    return descriptions.get(state, "Unknown state")


# Example usage and testing
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    print("=== Conversation State Machine Test ===\n")
    
    # Test 1: Active user
    print("Test 1: Active user")
    state, count, ts = update_conversation_state(
        current_state="ACTIVE",
        user_message="show me chargers",
        last_user_timestamp=datetime.now(timezone.utc),
        followup_count=0
    )
    print(f"Result: {state}, follow-ups: {count}")
    print(f"Description: {get_state_description(state)}\n")
    
    # Test 2: User says "let me think"
    print("Test 2: User says 'let me think'")
    state, count, ts = update_conversation_state(
        current_state="ACTIVE",
        user_message="let me think about it",
        last_user_timestamp=datetime.now(timezone.utc),
        followup_count=0
    )
    print(f"Result: {state}, follow-ups: {count}")
    print(f"Description: {get_state_description(state)}\n")
    
    # Test 3: 2 hours of silence
    print("Test 3: 2 hours of silence")
    two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2, minutes=5)
    state, count, ts = update_conversation_state(
        current_state="ACTIVE",
        user_message=None,
        last_user_timestamp=two_hours_ago,
        followup_count=0
    )
    print(f"Result: {state}, follow-ups: {count}")
    print(f"Follow-up message: {get_followup_message(state, 'Tata Curvv')}\n")
    
    # Test 4: User says "not interested"
    print("Test 4: User says 'not interested'")
    state, count, ts = update_conversation_state(
        current_state="ACTIVE",
        user_message="not interested, thanks",
        last_user_timestamp=datetime.now(timezone.utc),
        followup_count=1
    )
    print(f"Result: {state}, follow-ups: {count}")
    print(f"Description: {get_state_description(state)}\n")
    
    print("=== State Transition Map ===")
    for state, transitions in STATE_TRANSITIONS.items():
        print(f"\n{state}:")
        for trigger, result in transitions.items():
            print(f"  → {trigger}: {result}")