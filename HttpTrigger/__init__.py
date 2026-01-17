"""
Azure Function: RAG-powered chatbot
FINAL CLEAN VERSION
"""

import azure.functions as func
import logging
import os
import json
import sys
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import threading

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import AppConfig
from shared.prompt_manager import extract_vehicle_info
from shared.query_pipeline import answer_question
from shared.conversation_state import (
    update_conversation_state,
    should_send_followup,
    get_followup_message,
    get_state_description
)
from shared.key_facts import KeyFacts, update_facts_from_new_message

# Azure Table Storage
try:
    from azure.data.tables import TableServiceClient
    TABLES_AVAILABLE = True
except ImportError:
    TABLES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}

# --------------------------------------------------
# RATE LIMITING
# --------------------------------------------------
class RateLimiter:
    """Simple in-memory rate limiter for Azure Functions"""

    def __init__(self, max_requests_per_minute=60, max_requests_per_hour=500):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, client_id: str) -> tuple[bool, str]:
        """
        Check if request is allowed for client_id.
        Returns (allowed: bool, reason: str)
        """
        with self.lock:
            now = datetime.now(timezone.utc)
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)

            # Clean old entries
            if client_id in self.requests:
                self.requests[client_id] = [
                    ts for ts in self.requests[client_id] if ts > hour_ago
                ]

            # Count recent requests
            recent_requests = self.requests[client_id]
            minute_count = sum(1 for ts in recent_requests if ts > minute_ago)
            hour_count = len(recent_requests)

            # Check limits
            if minute_count >= self.max_requests_per_minute:
                return False, f"Rate limit exceeded: max {self.max_requests_per_minute} requests per minute"

            if hour_count >= self.max_requests_per_hour:
                return False, f"Rate limit exceeded: max {self.max_requests_per_hour} requests per hour"

            # Record this request
            self.requests[client_id].append(now)
            return True, ""

    def cleanup_old_entries(self):
        """Periodically cleanup old entries to prevent memory bloat"""
        with self.lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(hours=AppConfig.RATE_LIMIT_CLEANUP_HOURS)

            # Remove clients with no recent requests
            to_remove = []
            for client_id, timestamps in self.requests.items():
                if not timestamps or all(ts < cutoff for ts in timestamps):
                    to_remove.append(client_id)

            for client_id in to_remove:
                del self.requests[client_id]

            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} inactive rate limit entries")


# Initialize global rate limiter
rate_limiter = RateLimiter(
    max_requests_per_minute=AppConfig.RATE_LIMIT_PER_MINUTE,
    max_requests_per_hour=AppConfig.RATE_LIMIT_PER_HOUR
)

# --------------------------------------------------
# TABLE STORAGE
# --------------------------------------------------

def get_table_client():
    if not TABLES_AVAILABLE:
        return None

    conn_str = os.getenv("CHAT_STORAGE")
    if not conn_str:
        return None

    service = TableServiceClient.from_connection_string(conn_str)
    return service.get_table_client(AppConfig.CHAT_HISTORY_TABLE)


# --------------------------------------------------
# MEMORY HELPERS
# --------------------------------------------------

def load_user_memory(user_id: str, session_id: str):
    table = get_table_client()
    if not table:
        return [], "ACTIVE", 0, None, None, None, KeyFacts()

    try:
        entity = table.get_entity(user_id, session_id)

        history = json.loads(entity.get("conversation", "[]"))
        state = entity.get("conversation_state", "ACTIVE")
        followup_count = int(entity.get("followup_count", 0))

        last_user_ts = entity.get("last_user_timestamp")
        last_user_timestamp = (
            datetime.fromisoformat(last_user_ts).replace(tzinfo=timezone.utc)
            if last_user_ts else None
        )

        last_bot_message = entity.get("last_bot_message")

        bot_ts = entity.get("last_bot_timestamp")
        last_bot_timestamp = (
            datetime.fromisoformat(bot_ts).replace(tzinfo=timezone.utc)
            if bot_ts else None
        )

        # Load key_facts from storage
        key_facts_json = entity.get("key_facts", "")
        key_facts = KeyFacts.from_json(key_facts_json) if key_facts_json else KeyFacts()

        return (
            history,
            state,
            followup_count,
            last_user_timestamp,
            last_bot_message,
            last_bot_timestamp,
            key_facts
        )

    except Exception:
        return [], "ACTIVE", 0, None, None, None, KeyFacts()


def save_user_memory(
    user_id: str,
    session_id: str,
    history: List[Dict[str, str]],
    state: str,
    followup_count: int,
    last_user_timestamp: Optional[datetime],
    key_facts: Optional[KeyFacts] = None
):
    table = get_table_client()
    if not table:
        return

    last_bot_msg = None
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            last_bot_msg = msg.get("content")
            break

    entity = {
        "PartitionKey": user_id,
        "RowKey": session_id,
        "conversation": json.dumps(history),
        "conversation_state": state,
        "followup_count": followup_count,
        "last_user_timestamp": last_user_timestamp.isoformat() if last_user_timestamp else None,
        "last_bot_message": last_bot_msg,
        "last_bot_timestamp": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "key_facts": key_facts.to_json() if key_facts else ""
    }

    table.upsert_entity(entity)


def truncate_history(history: List[Dict[str, str]], max_messages: int = None):
    if max_messages is None:
        max_messages = AppConfig.MAX_STORED_MESSAGES
    return history[-max_messages:] if len(history) > max_messages else history


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main(req: func.HttpRequest) -> func.HttpResponse:

    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=200, headers=CORS_HEADERS)

    try:
        body = req.get_json() if req.get_body() else {}

        user_message = req.params.get("message") or body.get("message")
        user_id = req.params.get("user_id") or body.get("user_id") or "anonymous"
        session_id = req.params.get("session_id") or body.get("session_id")
        use_memory = (req.params.get("memory") or "true").lower() == "true"

        # Rate limiting check
        client_identifier = f"{user_id}:{req.headers.get('X-Forwarded-For', 'unknown')}"
        allowed, reason = rate_limiter.is_allowed(client_identifier)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_identifier}")
            return func.HttpResponse(
                json.dumps({
                    "error": "Rate limit exceeded",
                    "message": reason,
                    "retry_after": AppConfig.RATE_LIMIT_RETRY_AFTER
                }),
                status_code=429,
                headers=CORS_HEADERS
            )

        if not session_id:
            session_id = f"{user_id}:{datetime.now(timezone.utc).isoformat()}"

        if not user_message:
            return func.HttpResponse(
                json.dumps({"error": "Missing message"}),
                status_code=400,
                headers=CORS_HEADERS
            )

        # ----------------------------
        # Load memory
        # ----------------------------
        conversation_history = []
        state = "ACTIVE"
        followup_count = 0
        last_user_timestamp = None
        last_bot_message = None
        last_bot_timestamp = None
        key_facts = KeyFacts()

        if use_memory:
            (
                conversation_history,
                state,
                followup_count,
                last_user_timestamp,
                last_bot_message,
                last_bot_timestamp,
                key_facts
            ) = load_user_memory(user_id, session_id)

            conversation_history = truncate_history(conversation_history)

        # Update key_facts with new user message (lightweight regex extraction)
        key_facts = update_facts_from_new_message(user_message, key_facts)
        logger.info(f"Key facts: {key_facts.to_dict()}")

        # ----------------------------
        # Update state
        # ----------------------------
        state, followup_count, last_user_timestamp = update_conversation_state(
            state,
            user_message,
            last_user_timestamp,
            followup_count
        )

        # ----------------------------
        # Follow-up logic
        # ----------------------------
        time_since_bot = (
            datetime.now(timezone.utc) - last_bot_timestamp
            if last_bot_timestamp else None
        )

        if should_send_followup(state, last_bot_message, time_since_bot):
            followup_msg = get_followup_message(
                state,
                extract_vehicle_info(conversation_history).get("vehicle_name")
            )

            if followup_msg:
                save_user_memory(
                    user_id,
                    session_id,
                    conversation_history,
                    state,
                    followup_count,
                    last_user_timestamp,
                    key_facts
                )

                return func.HttpResponse(
                    json.dumps({"response": followup_msg}),
                    headers=CORS_HEADERS
                )

        # ----------------------------
        # LLM / RAG
        # ----------------------------
        try:
            result = answer_question(
                user_question=user_message,
                conversation_history=conversation_history,
                top_k=AppConfig.DEFAULT_RAG_TOP_K,
                key_facts=key_facts
            )
            bot_reply = result.answer

            # Update key_facts from bot response (may confirm facts)
            key_facts = update_facts_from_new_message(bot_reply, key_facts)

        except Exception as rag_error:
            logger.error(f"RAG pipeline error: {rag_error}", exc_info=True)

            # Graceful degradation - provide helpful error message
            bot_reply = (
                "I'm experiencing technical difficulties right now. "
                "Please try again in a moment. If the issue persists, "
                "contact us at support@zevpoint.com for assistance."
            )

            # Create minimal result for metadata
            from shared.query_pipeline import PipelineResult, QueryMetrics
            result = PipelineResult(
                answer=bot_reply,
                sources=[],
                metrics=QueryMetrics(),
                filters={},
                rewritten_queries=[],
                retrieved_chunks=[],
                intents=["general"],
                confidence_score=0.0,
                search_time_ms=0.0,
                llm_time_ms=0.0
            )

        # ----------------------------
        # Save conversation
        # ----------------------------
        if use_memory:
            conversation_history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": bot_reply}
            ])

            save_user_memory(
                user_id,
                session_id,
                conversation_history,
                state,
                followup_count,
                last_user_timestamp,
                key_facts
            )

        vehicle_info = extract_vehicle_info(conversation_history)

        # Get nudge from result (optional follow-up message)
        nudge = result.nudge if hasattr(result, 'nudge') else None
        # Get delayed nudge (shown after ~60 seconds of inactivity)
        delayed_nudge = result.delayed_nudge if hasattr(result, 'delayed_nudge') else None

        return func.HttpResponse(
            json.dumps({
                "response": bot_reply,
                "nudge": nudge,
                "delayed_nudge": delayed_nudge,
                "sources": result.sources if hasattr(result, 'sources') else [],
                "confidence": result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
                "metadata": {
                    "state": state,
                    "state_description": get_state_description(state),
                    "vehicle": vehicle_info.get("vehicle_name"),
                    "installation_type": vehicle_info.get("installation_type"),
                    "message_count": len(conversation_history),
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }),
            headers=CORS_HEADERS
        )

    except Exception as e:
        logger.error("HttpTrigger failed", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            headers=CORS_HEADERS
        )


# --------------------------------------------------
# CLEAR USER MEMORY
# --------------------------------------------------

def sanitize_odata_string(value: str) -> str:
    """
    Sanitize a string for use in OData filter queries.
    Escapes single quotes by doubling them (OData standard).
    """
    if not value:
        return value
    return value.replace("'", "''")


def clear_user_memory(user_id: str) -> bool:
    table = get_table_client()
    if not table:
        return False

    try:
        # Validate user_id format
        # Allow alphanumeric, underscore, hyphen, colon (for session IDs)
        if not user_id:
            logger.warning("Empty user_id provided")
            return False

        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-:")
        if not all(c in allowed_chars for c in user_id):
            logger.warning(f"Invalid user_id format: {user_id}")
            return False

        # Sanitize for OData query (escape single quotes)
        safe_user_id = sanitize_odata_string(user_id)

        entities = table.query_entities(query_filter=f"PartitionKey eq '{safe_user_id}'")
        for e in entities:
            table.delete_entity(e["PartitionKey"], e["RowKey"])
        return True
    except Exception as ex:
        logger.error(f"Failed to clear memory for user {user_id}: {ex}")
        return False
