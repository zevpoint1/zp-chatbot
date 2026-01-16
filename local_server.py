"""
Local Flask server for testing the chatbot UI
Wraps the Azure Function locally with session management
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
import uuid
from datetime import datetime, timedelta
from threading import Lock
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.query_pipeline import answer_question

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# In-memory session storage
# Format: {session_id: {"history": [...], "last_activity": datetime}}
session_store = {}
session_lock = Lock()

# Session cleanup configuration
MAX_SESSION_AGE = timedelta(hours=24)
MAX_SESSIONS = 1000

def cleanup_old_sessions():
    """Remove sessions older than MAX_SESSION_AGE"""
    with session_lock:
        now = datetime.now()
        expired_sessions = [
            sid for sid, data in session_store.items()
            if now - data["last_activity"] > MAX_SESSION_AGE
        ]
        for sid in expired_sessions:
            del session_store[sid]

        # If still too many sessions, remove oldest ones
        if len(session_store) > MAX_SESSIONS:
            sorted_sessions = sorted(
                session_store.items(),
                key=lambda x: x[1]["last_activity"]
            )
            to_remove = len(session_store) - MAX_SESSIONS
            for sid, _ in sorted_sessions[:to_remove]:
                del session_store[sid]

@app.route('/api/HttpTrigger', methods=['POST', 'OPTIONS'])
def chatbot():
    if request.method == 'OPTIONS':
        return '', 200

    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id')

        if not message:
            return jsonify({"error": "Missing message"}), 400

        # Generate session_id if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"Generated new session_id: {session_id}", file=sys.stderr)
        else:
            print(f"Using existing session_id: {session_id}", file=sys.stderr)

        # Cleanup old sessions periodically
        cleanup_old_sessions()

        # Get or create session
        with session_lock:
            if session_id not in session_store:
                session_store[session_id] = {
                    "history": [],
                    "last_activity": datetime.now()
                }

            session = session_store[session_id]
            conversation_history = session["history"]

            # Add user message to history
            conversation_history.append({
                "role": "user",
                "content": message
            })

        # Call the RAG pipeline with conversation history
        result = answer_question(
            message,
            conversation_history=conversation_history,
            top_k=8
        )

        # Add assistant response to history
        with session_lock:
            session_store[session_id]["history"].append({
                "role": "assistant",
                "content": result.answer
            })
            session_store[session_id]["last_activity"] = datetime.now()

        response_data = {
            "response": result.answer,
            "sources": result.sources,
            "confidence": result.confidence_score,
            "session_id": session_id
        }
        print(f"DEBUG: response_data keys: {list(response_data.keys())}", file=sys.stderr)
        print(f"DEBUG: session_id value: {response_data.get('session_id')}", file=sys.stderr)
        return jsonify(response_data)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting local chatbot server on http://localhost:5000")
    print("Open: http://localhost:8080/chatbot_ui/index.html in your browser")
    app.run(debug=False, host='0.0.0.0', port=5000)
