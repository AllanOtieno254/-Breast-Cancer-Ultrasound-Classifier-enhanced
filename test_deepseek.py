import os, requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv("DEEPSEEK_API_KEY")
BASE = "https://api.deepseek.com/v1"

headers = {
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"}
    ]
}

r = requests.post(f"{BASE}/chat/completions", headers=headers, json=payload)
print(r.status_code)
print(r.text)
