# Week 4: Multi-turn Context-Aware Agents - State Management

In Week 3, your Study Buddy gained the ability to use external tools and maintain short-term memory through conversation history. This week, we focus on **state management** to create a robust multi-turn conversational agent. State management ensures your agent tracks conversations, optimizes context, remembers key entities, and summarizes past interactions effectively. We’ll cover four components:

- **Conversation State Tracking**: Storing and managing the full conversation history.
- **Context Window Optimization**: Balancing recent and relevant context.
- **Entity-Aware Memory Systems**: Tracking specific details like names or topics.
- **Context Summarization Techniques**: Condensing history for efficiency.

This revision uses OpenAI’s API (via the provided proxy server) for all LLM tasks and introduces a conversation history approach that goes beyond retaining only the last `k` messages by implementing selective retrieval based on relevance and long-term storage.

---

## Conversation history management

### What is Conversation history management?

Conversation history management involves storing the entire conversation—user inputs and assistant responses—while intelligently retrieving relevant portions for context-aware responses. Unlike a fixed-size window (last `k` messages), this approach allows the agent to access older messages when relevant, improving coherence over long conversations.

### Why is it Important?

- **Long-term Coherence**: Enables references to earlier topics (e.g., “You asked about Python last week…”).
- **Flexibility**: Retrieves context dynamically based on query relevance, not just recency.
- **User Experience**: Feels more natural, mimicking human memory.

### How to Implement Conversation history management

1. **Full History Storage**: Store all messages in a list or JSON file for persistence.
2. **Selective Retrieval**: Use keyword matching or semantic similarity to fetch relevant past messages.
3. **Integration with OpenAI**: Include relevant history in the prompt sent to the API.
4. **Persistence**: Save history to a file to retain it across sessions.

### Example Implementation

Here’s an implementation that stores all messages and retrieves relevant ones using simple keyword matching:

```python
import json
import os
import requests
from datetime import datetime

HISTORY_FILE = "conversation_history.json"
conversation_history = []

def load_history():
    """Load conversation history from file."""
    global conversation_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            conversation_history = json.load(f)

def save_history():
    """Save conversation history to file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(conversation_history, f, indent=2)

def add_to_history(role, content):
    """Add a message to history with timestamp."""
    conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    save_history()

def get_relevant_history(query, max_messages=5):
    """Retrieve relevant history based on keyword matching."""
    query_words = set(query.lower().split())
    scored_messages = []
    
    for msg in conversation_history:
        msg_words = set(msg["content"].lower().split())
        overlap = len(query_words & msg_words)  # Count common words
        if overlap > 0:
            scored_messages.append((overlap, msg))
    
    # Sort by overlap score and return top messages
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in scored_messages[:max_messages]]

# Example usage
load_history()
add_to_history("user", "What is Python?")
add_to_history("assistant", "Python is a versatile programming language.")
add_to_history("user", "How does Java compare?")
add_to_history("assistant", "Java is more verbose but widely used in enterprise.")
relevant = get_relevant_history("Python vs Java")
print([msg["content"] for msg in relevant])
```

**Output**:
```
[
    "What is Python?",
    "Python is a versatile programming language.",
    "How does Java compare?",
    "Java is more verbose but widely used in enterprise."
]
```

### Notes
- **Persistence**: History is saved to `conversation_history.json`, allowing retention across sessions.
- **Relevance**: Keyword overlap is simple; for better results, use embeddings (covered later).
- **Scalability**: For large histories, consider a database like SQLite.

---

## Context Window Optimization

### What is Context Window Optimization?

Context window optimization involves selecting a subset of conversation history to include in the prompt, respecting OpenAI’s token limits (e.g., ~4096 tokens for `gpt-3.5-turbo`). This ensures the agent uses relevant context efficiently.

### Why is it Important?

- **Performance**: Reduces API costs and latency.
- **Relevance**: Focuses on context that matters for the current query.

### How to Implement Context Window Optimization

1. **Hybrid Approach**: Combine recent messages (last 3-5) with relevant older messages.
2. **Token Counting**: Estimate tokens to stay within limits.
3. **Dynamic Selection**: Use relevance scores to prioritize messages.

### Example Implementation

Here’s a hybrid approach using OpenAI’s API:

```python
API_URL = "http://socapi.deepaksilaych.me/student1"  # Your assigned URL
MAX_TOKENS = 3000  # Conservative limit for context

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 chars)."""
    return len(text) // 4

def get_optimized_context(query):
    """Combine recent and relevant history."""
    recent_msgs = conversation_history[-3:]  # Last 3 messages
    relevant_msgs = get_relevant_history(query, max_messages=3)
    # Combine and deduplicate
    context_msgs = list({msg["content"]: msg for msg in recent_msgs + relevant_msgs}.values())
    
    # Trim to fit token limit
    total_tokens = 0
    final_context = []
    for msg in context_msgs:
        msg_tokens = estimate_tokens(msg["content"])
        if total_tokens + msg_tokens < MAX_TOKENS:
            final_context.append(msg)
            total_tokens += msg_tokens
    return final_context

def call_openai(query, context):
    """Call OpenAI API with context."""
    headers = {"Content-Type": "application/json"}
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in context]
    messages.append({"role": "user", "content": query})
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "API error."

# Example usage
add_to_history("user", "What’s AI?")
add_to_history("assistant", "AI is artificial intelligence.")
context = get_optimized_context("Tell me more about AI.")
response = call_openai("Tell me more about AI.", context)
print(response)
```

**Output** (example):
```
AI involves machines mimicking human intelligence, like learning and problem-solving.
```

### Notes
- **Token Limit**: Adjust `MAX_TOKENS` based on API limits and query needs.
- **Enhancement**: Use `tiktoken` for precise token counting (`pip install tiktoken`).

---

## Entity-Aware Memory Systems

### What are Entity-Aware Memory Systems?

Entity-aware memory systems track specific entities (e.g., names, topics) across the conversation, storing them for precise recall. This enhances personalization and accuracy.

### Why is it Important?

- **Personalization**: References user-specific details (e.g., “You asked about AI earlier…”).
- **Accuracy**: Ensures consistent entity references.

### How to Implement Entity-Aware Memory Systems

1. **Entity Extraction**: Use keyword rules or embeddings for entity detection.
2. **Storage**: Maintain a dictionary mapping entities to contexts.
3. **Integration**: Include entity context in OpenAI prompts.

### Example Implementation

Here’s a simple entity tracker:

```python
entities = {}

def extract_entities(text):
    """Store capitalized words as entities."""
    words = text.split()
    for word in words:
        if word.istitle() and len(word) > 2:
            entities[word] = {"context": text, "timestamp": datetime.now().isoformat()}

def get_entity_context(entity):
    """Retrieve entity context if mentioned."""
    return entities.get(entity, {}).get("context", "")

# Example usage
add_to_history("user", "Tell me about Python.")
extract_entities("Tell me about Python.")
context = get_optimized_context("What’s Python used for?")
entity_context = get_entity_context("Python")
if entity_context:
    context.append({"role": "system", "content": f"Note: User previously mentioned Python: {entity_context}"})
response = call_openai("What’s Python used for?", context)
print(response)
```

**Output** (example):
```
Python is used for web development, data science, and automation.
```

### Notes
- **Improvement**: Use `spacy` for robust entity extraction (`pip install spacy`).
- **Storage**: Consider a database for large-scale entity tracking.

---

## Context Summarization Techniques

### What are Context Summarization Techniques?

Context summarization condenses conversation history into a concise form, preserving key information. This reduces token usage and focuses the agent on critical details.

### Why is it Important?

- **Efficiency**: Minimizes prompt size for faster API calls.
- **Clarity**: Improves response quality by highlighting relevant context.

### How to Implement Context Summarization Techniques

1. **Abstractive Summarization**: Use OpenAI to generate a summary.
2. **Selective Inclusion**: Combine summary with key messages.
3. **Prompt Design**: Instruct OpenAI to summarize accurately.

### Example Implementation

Here’s an abstractive summarization using OpenAI:

```python
def summarize_context():
    """Summarize conversation history using OpenAI."""
    full_context = " ".join([msg["content"] for msg in conversation_history])
    if estimate_tokens(full_context) < 50:
        return full_context
    
    summary_prompt = f"Summarize the following conversation concisely:\n{full_context}"
    summary_context = [{"role": "user", "content": summary_prompt}]
    summary = call_openai(summary_prompt, summary_context)
    return summary

# Example usage
add_to_history("user", "What is machine learning?")
add_to_history("assistant", "Machine learning is a subset of AI.")
add_to_history("user", "How does it work?")
add_to_history("assistant", "It uses algorithms to find patterns.")
summary = summarize_context()
print(summary)
```

**Output** (example):
```
Machine learning, a subset of AI, uses algorithms to identify patterns.
```

### Notes
- **Prompt Tuning**: Adjust `summary_prompt` for brevity or detail.
- **Fallback**: Use full context if too short to summarize.

---

## Flow Architecture

1. **User Input**: Add to `conversation_history` and save to file.
2. **Entity Extraction**: Update `entities` with detected entities.
3. **Context Selection**: Retrieve recent and relevant messages via `get_optimized_context`.
4. **Summarization**: Generate summary if history is long via `summarize_context`.
5. **OpenAI Call**: Send query with context to API.
6. **Response**: Store and display response.

### Flow Diagram

```
[User Input] → [Add to History] → [Extract Entities] → [Select Context] → [Summarize Context] → [OpenAI API Call] → [Output]
```

---

## Example Integration

Here’s the complete chatbot:

```python
import json
import os
import requests
from datetime import datetime

API_URL = "http://socapi.deepaksilaych.me/student1"
HISTORY_FILE = "conversation_history.json"
conversation_history = []
entities = {}
MAX_TOKENS = 3000

def load_history():
    global conversation_history
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            conversation_history = json.load(f)

def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(conversation_history, f, indent=2)

def add_to_history(role, content):
    conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    save_history()

def estimate_tokens(text):
    return len(text) // 4

def get_relevant_history(query, max_messages=5):
    query_words = set(query.lower().split())
    scored_messages = []
    for msg in conversation_history:
        msg_words = set(msg["content"].lower().split())
        overlap = len(query_words & msg_words)
        if overlap > 0:
            scored_messages.append((overlap, msg))
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in scored_messages[:max_messages]]

def get_optimized_context(query):
    recent_msgs = conversation_history[-3:]
    relevant_msgs = get_relevant_history(query, max_messages=3)
    context_msgs = list({msg["content"]: msg for msg in recent_msgs + relevant_msgs}.values())
    total_tokens = 0
    final_context = []
    for msg in context_msgs:
        msg_tokens = estimate_tokens(msg["content"])
        if total_tokens + msg_tokens < MAX_TOKENS:
            final_context.append(msg)
            total_tokens += msg_tokens
    return final_context

def extract_entities(text):
    words = text.split()
    for word in words:
        if word.istitle() and len(word) > 2:
            entities[word] = {"context": text, "timestamp": datetime.now().isoformat()}

def get_entity_context(entity):
    return entities.get(entity, {}).get("context", "")

def summarize_context():
    full_context = " ".join([msg["content"] for msg in conversation_history])
    if estimate_tokens(full_context) < 50:
        return full_context
    summary_prompt = f"Summarize the following conversation concisely:\n{full_context}"
    summary_context = [{"role": "user", "content": summary_prompt}]
    return call_openai(summary_prompt, summary_context)

def call_openai(query, context):
    headers = {"Content-Type": "application/json"}
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in context]
    messages.append({"role": "user", "content": query})
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "API error."

def run_chatbot():
    load_history()
    print("Welcome to the Study Buddy! Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        add_to_history("user", query)
        extract_entities(query)
        context = get_optimized_context(query)
        entity_context = get_entity_context(query.split()[0])
        if entity_context:
            context.append({"role": "system", "content": f"Note: User mentioned {query.split()[0]}: {entity_context}"})
        if len(conversation_history) > 10:
            summary = summarize_context()
            context.append({"role": "system", "content": f"Summary: {summary}"})
        response = call_openai(query, context)
        add_to_history("assistant", response)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    run_chatbot()
```

### Test Case
- **Input**: “What is AI?” → “Tell me more about AI.” → “How does it relate to Python?”
- **Output** (example):
  - “AI is artificial intelligence, enabling machines to mimic human thinking.”
  - “AI includes subfields like machine learning and natural language processing.”
  - “Python is widely used in AI for libraries like TensorFlow and scikit-learn.”
