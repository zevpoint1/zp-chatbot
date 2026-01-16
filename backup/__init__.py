import azure.functions as func
import logging
import os
import json
from openai import OpenAI, OpenAIError
# ----------------------------
# MEMORY STORAGE HELPERS
# ----------------------------

from azure.data.tables import TableServiceClient

def get_table_client():
    conn_str = os.getenv("CHAT_STORAGE")
    table_name = "ChatHistory"

    service = TableServiceClient.from_connection_string(conn_str)
    return service.get_table_client(table_name)

def load_user_memory(phone_number):
    table = get_table_client()
    try:
        entity = table.get_entity(partition_key="chat", row_key=phone_number)
        return json.loads(entity["conversation"])
    except:
        return []  # No conversation exists yet

def save_user_memory(phone_number, history):
    table = get_table_client()
    entity = {
        "PartitionKey": "chat",
        "RowKey": phone_number,
        "conversation": json.dumps(history)
    }
    table.upsert_entity(entity)

# CORS headers
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}

def load_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error loading file {path}: {e}")
        return ""

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Azure HTTP trigger function processed a request.")

    # Handle CORS preflight
    if req.method == "OPTIONS":
        return func.HttpResponse("", status_code=200, headers=CORS_HEADERS)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return func.HttpResponse("Missing OPENAI_API_KEY.", status_code=500, headers=CORS_HEADERS)

    client = OpenAI(api_key=api_key)

    # Parse user message
    user_message = req.params.get("message")
    if not user_message:
        try:
            user_message = req.get_json().get("message")
        except:
            user_message = None

    if not user_message:
        return func.HttpResponse("Please provide a message.", status_code=400, headers=CORS_HEADERS)

    # Load system prompt + knowledge
    system_prompt = load_file("system_prompt.txt")
    knowledge_base = load_file("knowledge_base.txt")

    combined_prompt = f"""
{system_prompt}

Here is the verified information you MUST use:
{knowledge_base}
"""

    # Call OpenAI chat API
    try:
        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "user", "content": user_message}
        ]

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.1-chat-latest"),
            messages=messages,
            max_completion_tokens=1024
        )

        bot_reply = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                bot_reply = choice.message.content
            elif hasattr(choice, 'text'):
                bot_reply = choice.text

        return func.HttpResponse(
            json.dumps({"response": bot_reply}),
            mimetype="application/json",
            headers=CORS_HEADERS
        )

    except OpenAIError as e:
        logging.error(f"OpenAI error: {e}")
        return func.HttpResponse(
            f"OpenAI error: {str(e)}",
            status_code=500,
            headers=CORS_HEADERS
        )
    except Exception as e:
        logging.error(f"Error calling OpenAI: {e}")
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500,
            headers=CORS_HEADERS
        )
