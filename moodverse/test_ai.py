import os, httpx
key = os.environ["GEMINI_API_KEY"]
model = os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it")
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
r = httpx.post(url, json={"contents": [{"parts": [{"text": "Reply with only this JSON: {\"ok\": true}"}]}]}, timeout=60)
print(r.status_code)
print(r.text[:600])