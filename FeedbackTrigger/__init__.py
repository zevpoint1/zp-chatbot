"""
Azure Function: Feedback collection endpoint
Allows tracking good/bad responses to improve chatbot over time
"""

import azure.functions as func
import logging
import os
import json
from datetime import datetime, timezone

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


def get_feedback_table_client():
    """Get Azure Table Storage client for feedback"""
    if not TABLES_AVAILABLE:
        return None

    conn_str = os.getenv("CHAT_STORAGE")
    if not conn_str:
        return None

    service = TableServiceClient.from_connection_string(conn_str)
    return service.get_table_client("ChatFeedback")


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    POST /api/FeedbackTrigger
    Body: {
        "user_id": "user123",
        "session_id": "session456",
        "message_id": "msg789",  # Optional: timestamp or unique ID of the bot message
        "user_question": "What is the price?",
        "bot_response": "The Aveo Pro costs Rs. 22,999...",
        "rating": "good" | "bad",  # Simple good/bad rating
        "issue_type": "irrelevant" | "wrong_info" | "markdown" | "language" | "other",  # Optional
        "notes": "User's additional feedback"  # Optional
    }
    """

    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=200, headers=CORS_HEADERS)

    try:
        body = req.get_json() if req.get_body() else {}

        user_id = body.get("user_id", "anonymous")
        session_id = body.get("session_id", "")
        user_question = body.get("user_question", "")
        bot_response = body.get("bot_response", "")
        rating = body.get("rating", "")
        issue_type = body.get("issue_type", "")
        notes = body.get("notes", "")

        # Validation
        if not rating or rating not in ["good", "bad"]:
            return func.HttpResponse(
                json.dumps({"error": "Rating must be 'good' or 'bad'"}),
                status_code=400,
                headers=CORS_HEADERS
            )

        if not user_question or not bot_response:
            return func.HttpResponse(
                json.dumps({"error": "Missing user_question or bot_response"}),
                status_code=400,
                headers=CORS_HEADERS
            )

        # Save to Table Storage
        table = get_feedback_table_client()
        if not table:
            logger.warning("Table Storage not available - feedback not saved")
            return func.HttpResponse(
                json.dumps({"message": "Feedback received but storage unavailable"}),
                status_code=200,
                headers=CORS_HEADERS
            )

        timestamp = datetime.now(timezone.utc)
        feedback_id = f"{user_id}:{timestamp.isoformat()}"

        entity = {
            "PartitionKey": user_id,
            "RowKey": timestamp.strftime("%Y%m%d%H%M%S%f"),  # Sortable timestamp
            "session_id": session_id,
            "user_question": user_question,
            "bot_response": bot_response,
            "rating": rating,
            "issue_type": issue_type,
            "notes": notes,
            "timestamp": timestamp.isoformat(),
            "created_at": timestamp.isoformat()
        }

        table.upsert_entity(entity)
        logger.info(f"Feedback saved: {rating} from {user_id}")

        return func.HttpResponse(
            json.dumps({
                "message": "Feedback saved successfully",
                "feedback_id": feedback_id
            }),
            status_code=200,
            headers=CORS_HEADERS
        )

    except Exception as e:
        logger.error(f"Feedback trigger failed: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            headers=CORS_HEADERS
        )
