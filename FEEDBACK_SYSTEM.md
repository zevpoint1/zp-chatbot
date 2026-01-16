# Chatbot Feedback System

## Overview

The feedback system allows you to track good and bad chatbot responses to identify patterns and improve response quality over time.

## Components

### 1. Feedback API Endpoint
**URL:** `https://zp-chatbot-v2.azurewebsites.net/api/feedback`

**Method:** POST

**Request Body:**
```json
{
  "user_id": "user123",
  "session_id": "session456",
  "message_id": "msg789",
  "user_question": "What is the price?",
  "bot_response": "The Aveo Pro costs Rs. 22,999...",
  "rating": "good",  // or "bad"
  "issue_type": "irrelevant",  // Optional: "irrelevant", "wrong_info", "markdown", "language", "missing_context", "other"
  "notes": "Response didn't mention installation"  // Optional
}
```

**Response:**
```json
{
  "message": "Feedback saved successfully",
  "feedback_id": "user123:2026-01-16T10:30:00"
}
```

### 2. Feedback Viewer Script
**File:** `view_feedback.py`

**Usage:**
```bash
# View all feedback
python view_feedback.py

# View only bad feedback
python view_feedback.py --rating bad

# View feedback from last 7 days
python view_feedback.py --days 7

# Show full question/response text
python view_feedback.py --full

# Analyze patterns in bad responses
python view_feedback.py --analyze

# Export to JSON
python view_feedback.py --export feedback.json

# Combine options
python view_feedback.py --rating bad --days 7 --analyze
```

### 3. Integration Example
**File:** `feedback_integration_example.html`

Shows how to add 👍/👎 buttons to your chatbot interface and submit feedback to the API.

## Storage

Feedback is stored in Azure Table Storage in the `ChatFeedback` table with this structure:

| Field | Type | Description |
|-------|------|-------------|
| PartitionKey | string | User ID |
| RowKey | string | Timestamp (sortable) |
| session_id | string | Chat session ID |
| user_question | string | User's question |
| bot_response | string | Bot's response |
| rating | string | "good" or "bad" |
| issue_type | string | Category of issue (for bad ratings) |
| notes | string | Additional feedback text |
| timestamp | string | ISO timestamp |

## How to Use

### Step 1: Deploy the Feedback Endpoint

The feedback endpoint will be deployed automatically when you deploy the chatbot:

```bash
func azure functionapp publish zp-chatbot-v2 --python
```

### Step 2: Add Feedback Buttons to Your UI

Add thumbs up/down buttons after each bot message. See `feedback_integration_example.html` for reference.

Key points:
- Add feedback buttons after each bot message
- Store the question/response context with a unique message ID
- On "bad" rating, show a form to collect issue type and notes
- Submit feedback to the API endpoint

### Step 3: Monitor Feedback

Use the `view_feedback.py` script to review feedback:

```bash
# Check for bad responses daily
python view_feedback.py --rating bad --days 1

# Weekly analysis
python view_feedback.py --days 7 --analyze
```

### Step 4: Use Insights to Improve

The analysis will show:
- **Common issue types** (irrelevant, wrong_info, markdown, etc.)
- **Sample bad responses** for each issue type
- **Patterns** in what users find unhelpful

Use this to:
1. **Identify prompt improvements** - If "irrelevant" is common, strengthen grounding instructions
2. **Find missing data** - If "wrong_info" is frequent, check if documents are incomplete
3. **Fix formatting issues** - If "markdown" appears, strengthen anti-markdown rules
4. **Track improvement** - Compare bad response rate week-over-week

## Issue Types

When users mark a response as "bad", they can specify:

- **irrelevant** - Response doesn't address the question
- **wrong_info** - Factually incorrect information
- **markdown** - Formatting issues (stars, asterisks)
- **language** - Wrong language or language mixing
- **missing_context** - Didn't use available documentation
- **other** - Other issues (user can add notes)

## Example Analysis Output

```
================================================================================
BAD RESPONSE ANALYSIS (23 entries)
================================================================================

COMMON ISSUES:
  irrelevant: 12 (52.2%)
  missing_context: 7 (30.4%)
  markdown: 3 (13.0%)
  wrong_info: 1 (4.3%)

--------------------------------------------------------------------------------
SAMPLE BAD RESPONSES BY TYPE:

IRRELEVANT:
  Q: What is the price?
  A: Zevpoint offers high-quality EV chargers for home and commercial use. We have various models to suit your needs...
  Note: Didn't answer the question

  Q: Which charger for Ola scooter?
  A: The Aveo X1 is a great choice for your needs
  Note: Recommended 4-wheeler charger for 2-wheeler

MISSING_CONTEXT:
  Q: How much does installation cost?
  A: Installation costs vary. Please contact us for details.
  Note: Pricing is in the docs but wasn't used
```

## Benefits

1. **Track improvement** - See if changes are working
2. **Find blind spots** - Discover what the bot struggles with
3. **Data-driven decisions** - Make improvements based on real user feedback
4. **Quality metrics** - Monitor good/bad ratio over time
5. **User insights** - Understand what users value in responses

## Privacy Note

- Store only necessary data for improvement
- User IDs are anonymous by default
- You can periodically clean old feedback data
- Feedback is only accessible to you via Azure Table Storage

## Next Steps

1. Deploy the feedback endpoint
2. Add feedback buttons to your chatbot UI
3. Start collecting feedback
4. Review weekly using `view_feedback.py --analyze`
5. Make improvements based on patterns
6. Track if bad response rate decreases
