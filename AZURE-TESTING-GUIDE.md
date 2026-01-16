# Testing Frontend with Azure Function

## Quick Setup (3 Steps)

### Step 1: Get Your Azure Function URL

1. Go to **Azure Portal**: https://portal.azure.com
2. Navigate to your **Function App**
3. Click on **HttpTrigger** (your function)
4. Click **"Get Function URL"** button
5. Copy the entire URL

**Your URL will look like:**
```
https://your-app-name.azurewebsites.net/api/HttpTrigger?code=ABC123XYZ...
```

---

### Step 2: Update the HTML File

Open `C:\chatbot-final\index-azure.html` in a text editor and find line ~140:

**Change this:**
```javascript
const apiUrl = "https://YOUR-FUNCTION-APP.azurewebsites.net/api/HttpTrigger";
```

**To your actual URL:**
```javascript
const apiUrl = "https://your-app-name.azurewebsites.net/api/HttpTrigger?code=ABC123XYZ...";
```

**Or** if you want to make it public (no authentication), remove the `?code=` part:
```javascript
const apiUrl = "https://your-app-name.azurewebsites.net/api/HttpTrigger";
```

---

### Step 3: Configure Azure CORS

For the frontend to connect to Azure, you MUST enable CORS:

1. In Azure Portal, go to your **Function App**
2. Click **CORS** (under API section in left menu)
3. Add these origins:
   ```
   http://localhost:8080
   file://
   *
   ```
   Or if you want to be specific:
   - `http://localhost:8080` (for local testing)
   - `https://zevpoint1.github.io` (for GitHub Pages)

4. Click **Save**
5. Wait 1 minute for changes to apply

---

### Step 4: Test Locally

1. **Open the file** in your browser:
   - Navigate to: `C:\chatbot-final\index-azure.html`
   - Right-click → Open with → Chrome/Edge/Firefox

   Or use a simple HTTP server:
   ```bash
   cd C:\chatbot-final
   python -m http.server 8080
   ```
   Then open: http://localhost:8080/index-azure.html

2. **Open Browser Console** (F12 → Console tab)

3. **Send a test message**: "I need a charger for Kia EV6"

4. **Check the console** for:
   - ✅ "Sending to Azure: https://your-app.azurewebsites.net..."
   - ✅ "Azure Response: { response: '...' }"
   - ✅ Green status bar: "Message sent"

---

## Troubleshooting

### Error: "Cannot connect to Azure"

**Cause**: CORS not configured

**Fix**:
1. Azure Portal → Function App → CORS
2. Add `http://localhost:8080` and `*`
3. Save and wait 1 minute

---

### Error: "HTTP 401 Unauthorized"

**Cause**: Function key is wrong or missing

**Fix**:
1. Get the correct function URL with `?code=` parameter
2. Or disable authentication:
   - Azure Portal → HttpTrigger → Integration
   - Change Authorization level to **Anonymous**
   - Save

---

### Error: "HTTP 404 Not Found"

**Cause**: Function not deployed or wrong URL

**Fix**:
1. Verify your Function App is running (Azure Portal → Overview)
2. Check the function name is correct (should be `HttpTrigger`)
3. Try accessing the URL directly in browser to test

---

### Bot doesn't respond / No error

**Cause**: API URL not updated in HTML

**Fix**:
1. Edit `index-azure.html`
2. Find line ~140
3. Replace `YOUR-FUNCTION-APP` with your actual function app name

---

### Error: "Failed to fetch"

**Causes**:
1. CORS not enabled
2. Azure Function is stopped
3. Network/firewall blocking request

**Fix**:
1. Check CORS settings
2. Verify function is running in Azure Portal
3. Try accessing function URL directly in browser

---

## Verify Azure Function is Working

### Test the function directly:

1. **Using Browser**:
   - Open this URL in browser:
     ```
     https://your-app.azurewebsites.net/api/HttpTrigger?message=test&user_id=test
     ```
   - You should see JSON response

2. **Using PowerShell**:
   ```powershell
   $body = @{
       message = "test"
       user_id = "test"
   } | ConvertTo-Json

   Invoke-RestMethod -Uri "https://your-app.azurewebsites.net/api/HttpTrigger" -Method POST -Body $body -ContentType "application/json"
   ```

3. **Using curl**:
   ```bash
   curl -X POST https://your-app.azurewebsites.net/api/HttpTrigger \
     -H "Content-Type: application/json" \
     -d '{"message":"test","user_id":"test"}'
   ```

---

## Response Format

**Expected Azure Response**:
```json
{
  "response": "Which electric vehicle do you drive?",
  "sources": ["..."],
  "confidence": 0.5,
  "metadata": {
    "state": "ACTIVE",
    "session_id": "web_user:2026-01-16T...",
    "vehicle": null,
    "message_count": 2,
    "timestamp": "2026-01-16T..."
  }
}
```

---

## Comparison: Local vs Azure

| Feature | Local Server | Azure Function |
|---------|-------------|----------------|
| **URL** | `http://localhost:5000/api/HttpTrigger` | `https://your-app.azurewebsites.net/api/HttpTrigger` |
| **Session Storage** | In-memory (RAM) | Azure Table Storage |
| **Persistence** | Lost on restart | Permanent |
| **Response Format** | `{response, sources, confidence, session_id}` | `{response, sources, confidence, metadata}` |
| **CORS** | Enabled for all | Must configure in Azure |
| **Authentication** | None | Function key (optional) |

---

## Checklist

Before testing with Azure:

- [ ] Azure Function is deployed and running
- [ ] Got the Function URL from Azure Portal
- [ ] Updated `index-azure.html` with correct URL
- [ ] Configured CORS in Azure (added `http://localhost:8080` or `*`)
- [ ] Environment variables set in Azure (OPENAI_API_KEY, QDRANT_URL, etc.)
- [ ] Opened browser console (F12) to see logs
- [ ] Tested with a simple message

---

## Quick Test Command

Create a simple test file:

```html
<!-- test-azure.html -->
<!DOCTYPE html>
<html>
<body>
<button onclick="testAzure()">Test Azure Function</button>
<pre id="result"></pre>

<script>
async function testAzure() {
    const url = "https://YOUR-APP.azurewebsites.net/api/HttpTrigger";
    const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message: "test", user_id: "test"})
    });
    const data = await response.json();
    document.getElementById("result").textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
```

---

## Need Help?

1. Check browser console (F12) for detailed error messages
2. Check Azure Portal → Function App → Monitor → Logs
3. Verify all environment variables are set in Azure App Settings

---

## After Testing Works

Once everything works with `index-azure.html`:

1. You can replace `index.html` with the Azure version
2. Or keep both:
   - `index.html` → Local testing (localhost:5000)
   - `index-azure.html` → Azure testing (production)
3. Deploy `index-azure.html` to GitHub Pages (rename to `index.html`)
