"""
Azure Timer Function: Automated Follow-up System
FIXED: Proper imports from shared module
"""

import azure.functions as func
import logging
import os
import json
import sys
from datetime import datetime, timedelta, timezone

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from shared module
from shared.prompt_manager import extract_vehicle_info
from shared.conversation_state import (
    update_conversation_state,
    should_send_followup,
    get_followup_message,
    get_state_description
)

# ... rest of your TimerFollowup code ...

# For Azure Table Storage
try:
    from azure.data.tables import TableServiceClient
    TABLES_AVAILABLE = True
except ImportError:
    TABLES_AVAILABLE = False
    logging.warning("azure.data.tables not available")

logger = logging.getLogger(__name__)


def get_table_client():
    """Get Azure Table Storage client for conversation history"""
    if not TABLES_AVAILABLE:
        return None
    
    conn_str = os.getenv("CHAT_STORAGE")
    if not conn_str:
        logger.warning("CHAT_STORAGE not configured")
        return None
    
    table_name = "ChatHistory"
    try:
        service = TableServiceClient.from_connection_string(conn_str)
        return service.get_table_client(table_name)
    except Exception as e:
        logger.error("Failed to connect to Table Storage: %s", str(e))
        return None


def send_message_to_user(user_id: str, message: str, channel: str = "web") -> bool:
    """
    Send message to user via their preferred channel.
    
    Args:
        user_id: User identifier
        message: Message to send
        channel: Communication channel (web, sms, email, whatsapp)
    
    Returns:
        bool: True if sent successfully
    
    TODO: Implement actual message sending:
    - Web: Store in pending messages or push notification
    - SMS: Twilio, AWS SNS, or other SMS service
    - Email: SendGrid, AWS SES, or other email service
    - WhatsApp: Twilio WhatsApp API
    
    Example implementations:
    
    # SMS via Twilio:
    # from twilio.rest import Client
    # client = Client(account_sid, auth_token)
    # client.messages.create(
    #     to=user_phone,
    #     from_=twilio_number,
    #     body=message
    # )
    
    # Email via SendGrid:
    # from sendgrid import SendGridAPIClient
    # from sendgrid.helpers.mail import Mail
    # sg = SendGridAPIClient(api_key)
    # mail = Mail(from_email, to_email, subject, message)
    # sg.send(mail)
    """
    msg_preview = message[:50] if len(message) > 50 else message
    logger.info("[%s] Sending to %s: %s...", channel.upper(), user_id, msg_preview)
    
    # PLACEHOLDER: Just log for now
    # Replace this with actual message sending
    logger.info("Follow-up message queued: %s", message)
    
    # For web channel, you might store in a "pending_messages" table
    # that gets retrieved when user next opens the chat
    if channel == "web":
        # TODO: Store message for delivery when user reconnects
        # store_pending_message(user_id, message)
        return True
    
    return True


def main(mytimer: func.TimerRequest) -> None:
    if mytimer.past_due:
        logger.info("Timer is running late")

    logger.info("FOLLOW-UP TIMER STARTED")

    table = get_table_client()
    if not table:
        logger.error("Table storage unavailable")
        return

    try:
        entities = table.query_entities(
            query_filter="conversation_state ne 'CLOSED'"
        )

        for entity in entities:
            try:
                # ✅ Correct identity
                user_id = entity["PartitionKey"]
                session_id = entity["RowKey"]

                state = entity.get("conversation_state", "ACTIVE")
                followup_count = int(entity.get("followup_count", 0))

                history = json.loads(entity.get("conversation", "[]"))

                last_user_ts = entity.get("last_user_timestamp")
                if not last_user_ts:
                    continue

                last_user_timestamp = datetime.fromisoformat(last_user_ts)
                if last_user_timestamp.tzinfo is None:
                    last_user_timestamp = last_user_timestamp.replace(tzinfo=timezone.utc)

                last_bot_message = entity.get("last_bot_message")

                bot_ts = entity.get("last_bot_timestamp")
                last_bot_timestamp = (
                    datetime.fromisoformat(bot_ts).replace(tzinfo=timezone.utc)
                    if bot_ts else None
                )

                # 🔁 State update (time-based)
                new_state, new_count, _ = update_conversation_state(
                    current_state=state,
                    user_message=None,
                    last_user_timestamp=last_user_timestamp,
                    followup_count=followup_count
                )

                time_since_bot = (
                    datetime.now(timezone.utc) - last_bot_timestamp
                    if last_bot_timestamp else None
                )

                # ✅ Correct follow-up check
                if should_send_followup(new_state, last_bot_message, time_since_bot):
                    vehicle_info = extract_vehicle_info(history)
                    followup_msg = get_followup_message(
                        new_state,
                        vehicle_info.get("vehicle_name")
                    )

                    if followup_msg:
                        history.append({
                            "role": "assistant",
                            "content": followup_msg,
                            "type": "automated_followup",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })

                        entity["conversation"] = json.dumps(history)
                        entity["conversation_state"] = new_state
                        entity["followup_count"] = new_count
                        entity["last_bot_message"] = followup_msg
                        entity["last_bot_timestamp"] = datetime.now(timezone.utc).isoformat()
                        entity["updated_at"] = datetime.now(timezone.utc).isoformat()

                        table.upsert_entity(entity)

                        # ✅ Send to USER, not session
                        send_message_to_user(user_id, followup_msg, channel="web")

                # Update state even without follow-up
                elif new_state != state:
                    entity["conversation_state"] = new_state
                    entity["followup_count"] = new_count
                    entity["updated_at"] = datetime.now(timezone.utc).isoformat()
                    table.upsert_entity(entity)

            except Exception as e:
                logger.error("Session error %s:%s", user_id, session_id, exc_info=True)

    except Exception as e:
        logger.error("Timer failed", exc_info=True)
