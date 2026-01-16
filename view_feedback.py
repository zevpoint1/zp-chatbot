"""
Script to view and analyze chatbot feedback from Azure Table Storage
Usage: python view_feedback.py [--rating good|bad] [--days 7] [--export feedback.json]
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from collections import Counter
from dotenv import load_dotenv

try:
    from azure.data.tables import TableServiceClient
    TABLES_AVAILABLE = True
except ImportError:
    TABLES_AVAILABLE = False
    print("ERROR: azure-data-tables not installed. Run: pip install azure-data-tables")
    sys.exit(1)

load_dotenv()


def get_feedback_table_client():
    """Get Azure Table Storage client for feedback"""
    conn_str = os.getenv("CHAT_STORAGE")
    if not conn_str:
        print("ERROR: CHAT_STORAGE environment variable not set")
        sys.exit(1)

    service = TableServiceClient.from_connection_string(conn_str)
    return service.get_table_client("ChatFeedback")


def fetch_feedback(rating_filter=None, days=None):
    """Fetch feedback from Azure Table Storage"""
    table = get_feedback_table_client()

    try:
        # Build query filter
        filters = []

        if rating_filter:
            filters.append(f"rating eq '{rating_filter}'")

        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_str = cutoff.isoformat()
            filters.append(f"timestamp ge '{cutoff_str}'")

        query_filter = " and ".join(filters) if filters else None

        # Fetch entities
        entities = list(table.query_entities(query_filter=query_filter))

        return entities

    except Exception as e:
        print(f"ERROR fetching feedback: {e}")
        return []


def display_feedback(entities, show_full=False):
    """Display feedback in readable format"""
    if not entities:
        print("\nNo feedback found matching criteria.\n")
        return

    print(f"\n{'='*80}")
    print(f"FEEDBACK SUMMARY ({len(entities)} entries)")
    print(f"{'='*80}\n")

    # Statistics
    ratings = Counter(e.get("rating") for e in entities)
    issue_types = Counter(e.get("issue_type") for e in entities if e.get("issue_type"))

    print(f"RATINGS:")
    print(f"  Good: {ratings.get('good', 0)}")
    print(f"  Bad:  {ratings.get('bad', 0)}")

    if issue_types:
        print(f"\nISSUE TYPES (for bad ratings):")
        for issue, count in issue_types.most_common():
            print(f"  {issue}: {count}")

    print(f"\n{'-'*80}\n")

    # Individual entries
    for i, entity in enumerate(sorted(entities, key=lambda x: x.get("timestamp", ""), reverse=True), 1):
        rating = entity.get("rating", "unknown")
        timestamp = entity.get("timestamp", "")
        user_id = entity.get("PartitionKey", "")
        session_id = entity.get("session_id", "")
        issue_type = entity.get("issue_type", "")
        notes = entity.get("notes", "")

        print(f"[{i}] {rating.upper()} - {timestamp[:19]} - User: {user_id}")

        if issue_type:
            print(f"    Issue Type: {issue_type}")

        if notes:
            print(f"    Notes: {notes}")

        if show_full:
            question = entity.get("user_question", "")[:100]
            response = entity.get("bot_response", "")[:200]
            print(f"    Question: {question}...")
            print(f"    Response: {response}...")

        print()


def export_feedback(entities, output_file):
    """Export feedback to JSON file"""
    data = []
    for entity in entities:
        data.append({
            "timestamp": entity.get("timestamp"),
            "user_id": entity.get("PartitionKey"),
            "session_id": entity.get("session_id"),
            "rating": entity.get("rating"),
            "issue_type": entity.get("issue_type"),
            "notes": entity.get("notes"),
            "user_question": entity.get("user_question"),
            "bot_response": entity.get("bot_response")
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nExported {len(data)} feedback entries to {output_file}\n")


def analyze_bad_responses(entities):
    """Analyze patterns in bad responses"""
    bad_feedback = [e for e in entities if e.get("rating") == "bad"]

    if not bad_feedback:
        print("\nNo bad feedback to analyze.\n")
        return

    print(f"\n{'='*80}")
    print(f"BAD RESPONSE ANALYSIS ({len(bad_feedback)} entries)")
    print(f"{'='*80}\n")

    # Common issues
    issue_types = Counter(e.get("issue_type") for e in bad_feedback if e.get("issue_type"))

    print("COMMON ISSUES:")
    for issue, count in issue_types.most_common():
        print(f"  {issue}: {count} ({count/len(bad_feedback)*100:.1f}%)")

    # Sample bad responses by type
    print(f"\n{'-'*80}")
    print("SAMPLE BAD RESPONSES BY TYPE:\n")

    for issue_type in issue_types.keys():
        samples = [e for e in bad_feedback if e.get("issue_type") == issue_type][:2]

        print(f"\n{issue_type.upper()}:")
        for sample in samples:
            print(f"  Q: {sample.get('user_question', '')[:80]}")
            print(f"  A: {sample.get('bot_response', '')[:150]}...")
            if sample.get('notes'):
                print(f"  Note: {sample.get('notes')}")
            print()


def main():
    parser = argparse.ArgumentParser(description="View chatbot feedback from Azure Table Storage")
    parser.add_argument("--rating", choices=["good", "bad"], help="Filter by rating")
    parser.add_argument("--days", type=int, help="Show feedback from last N days")
    parser.add_argument("--export", help="Export to JSON file")
    parser.add_argument("--full", action="store_true", help="Show full question/response text")
    parser.add_argument("--analyze", action="store_true", help="Analyze bad responses")

    args = parser.parse_args()

    # Fetch feedback
    entities = fetch_feedback(rating_filter=args.rating, days=args.days)

    # Display or analyze
    if args.analyze:
        analyze_bad_responses(entities)
    else:
        display_feedback(entities, show_full=args.full)

    # Export if requested
    if args.export:
        export_feedback(entities, args.export)


if __name__ == "__main__":
    main()
