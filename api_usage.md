# Using the AI Agent API Server

This document explains how you can use the AI agent backend server that your mentor set up. The server provides an OpenAI-compatible interface with rate limits.
(IF YOU DONT KNOW WHAT I AM TALKING HERE, STUDY PROVIDED MATERIAL)
---

## 1. Using It as an API

You can call the server via HTTP requests, just like calling OpenAI API.

### Steps:

* Each student has their own API endpoint URL

* Send POST requests with the OpenAI Chat Completion payload.

### Example using Python `requests`:

```python
import requests

API_URL = "http://socapi.deepaksilaych.me/student1"  # Your assigned URL instead of openAI api

headers = {
    "Content-Type": "application/json"
}

payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Explain quantum computing simply."}
    ]
}

response = requests.post(API_URL, headers=headers, json=payload)

print(response.json())
```

---

## 2. Using It as a Python Library with `openai` Interface

You can use the official `openai` Python library interface but redirect its calls through the proxy server.

### Setup:

* Install OpenAI Python library if you don’t have it:

  ```
  pip install openai requests
  ```

* Use this monkey-patching code snippet before making calls:

save following in newopenai.py
```python
import openai
import requests

# Dummy key since the real API key is not used here
openai.api_key = "dummy"

# Your proxy URL (change this to your assigned student URL)
PROXY_URL = "http://socapi.deepaksilaych.me/student1"

def custom_chat_create(**kwargs):
    response = requests.post(PROXY_URL, json=kwargs)
    if response.status_code != 200:
        raise Exception(response.json())
    return response.json()

# Override the OpenAI chat create method to call your proxy
openai.ChatCompletion.create = custom_chat_create
```

### Example usage:

```python
from newopenai.key import openai
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "What is LangChain?"}
    ]
)

print(response['choices'][0]['message']['content'])
```

---

## Notes

* Each student has a **separate API endpoint and rate limit** (30 calls per week).
* Make sure to use your assigned URL to avoid hitting limits for others.
* If you get rate-limited, wait until your quota resets or let me know if you need to extend it
* The API follows the same format as OpenAI’s Chat Completion API, so you can reuse your OpenAI code with minimal changes.
