# Quick Start: Feedback System

## ✅ Deployed Successfully

Your feedback endpoint is live at:
**https://zp-chatbot-v2-ewapbch4bmcgh6eu.southindia-01.azurewebsites.net/api/feedback**

## 🎯 What You Can Do Now

### 1. Add Feedback Buttons to Your Chatbot UI

See `feedback_integration_example.html` for the full implementation. Basic example:

```html
<!-- Add these buttons after each bot message -->
<div class="feedback-buttons">
    <button onclick="submitFeedback('good')">👍 Helpful</button>
    <button onclick="submitFeedback('bad')">👎 Not Helpful</button>
</div>

<script>
async function submitFeedback(rating) {
    const response = await fetch('https://zp-chatbot-v2-ewapbch4bmcgh6eu.southindia-01.azurewebsites.net/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            user_id: 'user123',
            session_id: 'session456',
            user_question: 'What is the price?',
            bot_response: 'The Aveo Pro costs Rs. 22,999',
            rating: rating,  // 'good' or 'bad'
            issue_type: 'irrelevant',  // optional
            notes: 'User feedback'  // optional
        })
    });
}
</script>
```

### 2. View Feedback Data

```bash
# View all feedback
python view_feedback.py

# View only bad responses
python view_feedback.py --rating bad

# View last 7 days
python view_feedback.py --days 7

# Analyze patterns in bad responses
python view_feedback.py --analyze

# Export to JSON
python view_feedback.py --export feedback_data.json
```

### 3. Weekly Analysis Routine

Run this weekly to identify issues:

```bash
python view_feedback.py --rating bad --days 7 --analyze
```

This will show:
- **Common issue types** (irrelevant, wrong_info, markdown, etc.)
- **Sample bad responses** for each category
- **Percentage breakdown** of issues

### 4. Issue Types Available

When marking responses as "bad", users can specify:

- `irrelevant` - Response doesn't address the question
- `wrong_info` - Factually incorrect information
- `markdown` - Formatting issues (stars, asterisks)
- `language` - Wrong language or language mixing
- `missing_context` - Didn't use available documentation
- `other` - Other issues

## 📊 Example Analysis Output

```
BAD RESPONSE ANALYSIS (15 entries)
================================================================================

COMMON ISSUES:
  irrelevant: 8 (53.3%)
  missing_context: 5 (33.3%)
  markdown: 2 (13.3%)

SAMPLE BAD RESPONSES BY TYPE:

IRRELEVANT:
  Q: What is the price?
  A: We offer various chargers for different needs...
  Note: Didn't answer the question directly

MISSING_CONTEXT:
  Q: Does it work with Tata Nexon?
  A: Our chargers work with most EVs
  Note: Should have said "Yes" based on documentation
```

## 🔄 Improvement Cycle

1. **Collect** - Users provide feedback via thumbs up/down
2. **Analyze** - Weekly review using `view_feedback.py --analyze`
3. **Fix** - Update prompts or documentation based on patterns
4. **Deploy** - Push changes to Azure
5. **Monitor** - Track if bad response rate decreases

## 📁 Files Created

- `FeedbackTrigger/__init__.py` - Azure Function for collecting feedback
- `FeedbackTrigger/function.json` - Function configuration
- `view_feedback.py` - Script to view and analyze feedback
- `feedback_integration_example.html` - Example UI integration
- `FEEDBACK_SYSTEM.md` - Complete documentation

## 🗄️ Data Storage

Feedback is stored in Azure Table Storage (`ChatFeedback` table) with:
- User ID, session ID, timestamps
- User question and bot response
- Rating (good/bad)
- Issue type and notes
- All feedback is private and only accessible to you

## Next Steps

1. Add feedback buttons to your chatbot UI
2. Start collecting feedback from real users
3. Run weekly analysis to identify patterns
4. Make improvements based on data
5. Track if changes reduce bad response rate
