Below is the Week 3 material for the "Study Buddy Project" curriculum, designed to build on the Retrieval-Augmented Generation (RAG) system from Week 2. This week introduces students to integrating external tools (like Tavily, Wikipedia, and a calculator) and adding memory to their AI agents, enhancing their Study Buddy chatbot’s versatility and conversational coherence. The material follows the structure and depth of previous weeks, avoids frameworks like LangChain, and provides detailed explanations and code examples for manual implementation.

---

# Week 3: Tools and Memory in AI Agents

[⬅️ Back to Project Overview](../Readme.md)

---

## Index

- [Introduction](#introduction)
- [Part 1: External Tools](#part-1-external-tools)
  - [What Are External Tools?](#what-are-external-tools)
  - [Tools Overview](#tools-overview)
  - [Setting Up the Tools](#setting-up-the-tools)
  - [Integrating Tools with Your Agent](#integrating-tools-with-your-agent)
- [Part 2: Memory in AI Agents](#part-2-memory-in-ai-agents)
  - [What Is Memory?](#what-is-memory)
  - [Implementing Conversation History](#implementing-conversation-history)
- [Part 3: Combining Tools and Memory](#part-3-combining-tools-and-memory)
  - [Updating the Chatbot](#updating-the-chatbot)
  - [Full Code Example](#full-code-example)
- [Testing and Debugging](#testing-and-debugging)
- [Assignment](#assignment)
- [Resources](#resources)

---

## Introduction

In Week 2, you built a RAG-based Study Buddy chatbot that retrieves relevant information from your Markdown notes to answer questions. This week, we’ll expand its capabilities by adding **external tools** and **memory**. External tools allow your agent to fetch real-time information (e.g., via web searches with Tavily), look up facts (e.g., via Wikipedia), or perform calculations—tasks beyond the scope of your local notes. Memory enables your chatbot to remember previous interactions, making multi-turn conversations more natural and context-aware.

By the end of Week 3, your Study Buddy will be a more powerful assistant, capable of handling diverse queries and maintaining conversational context. Let’s dive in!

---

## Part 1: External Tools

### What Are External Tools?

External tools extend your AI agent’s abilities by connecting it to external resources or functions. While RAG retrieves information from a static knowledge base (your notes), tools enable dynamic interactions with the web, APIs, or computational resources. This makes your agent more versatile, allowing it to answer questions like “What’s the latest AI news?” or “What’s 15 times 23?”—queries your notes might not cover.

### Tools Overview

We’ll integrate three tools into your Study Buddy:

1. **Tavily**: A search engine API for retrieving up-to-date web information.
2. **Wikipedia**: An API for quick factual lookups from a vast knowledge repository.
3. **Calculator**: A simple function for evaluating mathematical expressions.

These tools complement RAG by addressing different types of queries:
- **Tavily**: For research or current events.
- **Wikipedia**: For general knowledge or definitions.
- **Calculator**: For numerical tasks.

### Setting Up the Tools

Here’s how to set up each tool manually:

#### 1. Tavily
- **Purpose**: Search the web for information not in your notes.
- **Setup**:
  - Sign up at [Tavily](https://tavily.com) to get an API key.
  - Install the `requests` library: `pip install requests`.
- **Code**:
  ```python
  import requests

  def search_tavily(query, api_key):
      url = "https://api.tavily.com/search"
      payload = {
          "api_key": api_key,
          "query": query,
          "search_depth": "basic",
          "include_answer": True
      }
      response = requests.post(url, json=payload)
      if response.status_code == 200:
          data = response.json()
          return data.get("answer", "No answer found.")
      return "Search failed."
  ```
- **Notes**: Store your API key securely (e.g., in a `.env` file) and avoid hardcoding it.

#### 2. Wikipedia
- **Purpose**: Retrieve concise factual summaries.
- **Setup**:
  - Install the `wikipedia` library: `pip install wikipedia`.
- **Code**:
  ```python
  import wikipedia

  def search_wikipedia(query):
      try:
          summary = wikipedia.summary(query, sentences=2)
          return summary
      except wikipedia.exceptions.DisambiguationError as e:
          return f"Multiple options found: {e.options}"
      except wikipedia.exceptions.PageError:
          return "No page found."
  ```
- **Notes**: Handles errors like ambiguous queries (e.g., “Apple” could mean the fruit or the company).

#### 3. Calculator
- **Purpose**: Perform basic arithmetic operations.
- **Setup**: No external libraries needed; we’ll use Python’s `ast` module for safe evaluation.
- **Code**:
  ```python
  import ast
  import operator

  def calculate(expression):
      try:
          node = ast.parse(expression, mode='eval')
          result = eval_(node.body)
          return str(result)
      except Exception:
          return "Calculation error."

  def eval_(node):
      if isinstance(node, ast.Num):
          return node.n
      elif isinstance(node, ast.BinOp):
          left = eval_(node.left)
          right = eval_(node.right)
          ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
          return ops[type(node.op)](left, right)
      else:
          raise ValueError("Unsupported expression")
  ```
- **Notes**: Safely evaluates expressions like “15 * 23” while preventing execution of malicious code.

### Integrating Tools with Your Agent

To use these tools, your agent needs to:
1. **Decide Which Tool to Use**: Implement simple keyword-based logic.
2. **Process Tool Outputs**: Integrate results into the chatbot’s response.

#### Tool Selection Logic
- **Approach**: Check the query for keywords to route it to the appropriate tool or fall back to RAG.
- **Code**:
  ```python
  def select_tool(query):
      query_lower = query.lower()
      if "search" in query_lower or "latest" in query_lower:
          return "tavily"
      elif "wiki" in query_lower or "who is" in query_lower:
          return "wikipedia"
      elif any(op in query_lower for op in ["+", "-", "*", "/", "calculate"]):
          return "calculator"
      else:
          return "rag"
  ```
- **Explanation**: 
  - “search” or “latest” triggers Tavily.
  - “wiki” or “who is” triggers Wikipedia.
  - Math operators or “calculate” trigger the calculator.
  - Default is RAG for note-based queries.

#### Handling Tool Outputs
Each tool returns a string that can be used directly as the chatbot’s response or passed to the LLM for further processing. We’ll use the tool output directly for simplicity, reserving the LLM for RAG responses.

---

## Part 2: Memory in AI Agents

### What Is Memory?

Memory in AI agents refers to the ability to retain and use information from previous interactions. Without memory, each query is treated in isolation, making multi-turn conversations disjointed (e.g., “What’s AI?” followed by “Tell me more” wouldn’t connect). Memory ensures context persists, improving coherence and user experience.

For the Study Buddy, we’ll implement **short-term memory** via conversation history, storing recent messages to inform future responses.

### Implementing Conversation History

- **Approach**: Store messages in a list and include them in the prompt for RAG queries.
- **Code**:
  ```python
  conversation_history = []

  def add_to_history(role, content):
      conversation_history.append({"role": role, "content": content})
      # Limit history to avoid token overflow (e.g., last 10 messages)
      if len(conversation_history) > 10:
          conversation_history.pop(0)

  def create_prompt_with_history(query):
      history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
      return f"{history_text}\nUser: {query}\nAssistant:"
  ```
- **Explanation**:
  - **Storage**: `conversation_history` is a list of dictionaries with `role` (user/assistant) and `content`.
  - **Truncation**: Limits history to 10 messages to fit GPT-2’s token limit (1024 tokens).
  - **Prompt**: Concatenates history with the current query, providing context.

---

## Part 3: Combining Tools and Memory

### Updating the Chatbot

Integrate tools and memory into your Week 2 RAG chatbot:
1. **Tool Handling**: Use `select_tool` to route queries and call the appropriate function.
2. **Memory**: Add each query and response to `conversation_history`, using it for RAG prompts.
3. **Response Generation**: Use tool outputs directly or generate RAG responses with history.

#### Main Query Handler
- **Code**:
  ```python
  def handle_query(query, generator, embedder, index, chunks, tavily_api_key):
      tool = select_tool(query)
      if tool == "tavily":
          result = search_tavily(query, tavily_api_key)
      elif tool == "wikipedia":
          result = search_wikipedia(query)
      elif tool == "calculator":
          expression = query.split("calculate", 1)[1].strip() if "calculate" in query.lower() else query
          result = calculate(expression)
      else:  # RAG
          relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
          prompt = create_prompt_with_history(query) + f"\nBased on the notes: {relevant_chunks}"
          response = generator(prompt, max_length=100, num_return_sequences=1)
          result = response[0]["generated_text"].split("Assistant:")[1].strip()
      add_to_history("user", query)
      add_to_history("assistant", result)
      return result
  ```

### Full Code Example

Here’s the complete Week 3 chatbot:

```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import wikipedia
import ast
import operator

# --- Tool Functions ---
def search_tavily(query, api_key):
    url = "https://api.tavily.com/search"
    payload = {"api_key": api_key, "query": query, "search_depth": "basic", "include_answer": True}
    response = requests.post(url, json=payload)
    return response.json().get("answer", "No answer found.") if response.status_code == 200 else "Search failed."

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No page found."

def calculate(expression):
    try:
        node = ast.parse(expression, mode='eval')
        return str(eval_(node.body))
    except Exception:
        return "Calculation error."

def eval_(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = eval_(node.left)
        right = eval_(node.right)
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
        return ops[type(node.op)](left, right)
    else:
        raise ValueError("Unsupported expression")

# --- RAG Functions ---
def load_and_chunk_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    chunks = text.split('\n\n')
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]

def generate_embeddings(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings, embedder

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(question, embedder, index, chunks, k=2):
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, k)
    return [chunks[idx] for idx in indices[0]]

# --- Memory Functions ---
conversation_history = []

def add_to_history(role, content):
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > 10:
        conversation_history.pop(0)

def create_prompt_with_history(query):
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
    return f"{history_text}\nUser: {query}\nAssistant:"

# --- Main Logic ---
def select_tool(query):
    query_lower = query.lower()
    if "search" in query_lower or "latest" in query_lower:
        return "tavily"
    elif "wiki" in query_lower or "who is" in query_lower:
        return "wikipedia"
    elif any(op in query_lower for op in ["+", "-", "*", "/", "calculate"]):
        return "calculator"
    else:
        return "rag"

def handle_query(query, generator, embedder, index, chunks, tavily_api_key):
    tool = select_tool(query)
    if tool == "tavily":
        result = search_tavily(query, tavily_api_key)
    elif tool == "wikipedia":
        result = search_wikipedia(query)
    elif tool == "calculator":
        expression = query.split("calculate", 1)[1].strip() if "calculate" in query.lower() else query
        result = calculate(expression)
    else:
        relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
        prompt = create_prompt_with_history(query) + f"\nBased on the notes: {relevant_chunks}"
        response = generator(prompt, max_length=100, num_return_sequences=1)
        result = response[0]["generated_text"].split("Assistant:")[1].strip()
    add_to_history("user", query)
    add_to_history("assistant", result)
    return result

# --- Chatbot Loop ---
def run_chatbot(file_path, tavily_api_key):
    print("Loading document and models...")
    chunks = load_and_chunk_document(file_path)
    embeddings, embedder = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)
    generator = pipeline("text-generation", model="gpt2")
    
    print("Welcome to the Study Buddy! Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        response = handle_query(query, generator, embedder, index, chunks, tavily_api_key)
        print(f"Assistant: {response}")

# Start the chatbot
TAVILY_API_KEY = "your-tavily-api-key-here"  # Replace with your key
run_chatbot("my_notes.md", TAVILY_API_KEY)
```
## This is a list of some toola you may find usefull: [Tools](https://github.com/GetStream/ai-agent-tools-catalog) 

#### Setup Instructions:
1. Install dependencies: `pip install transformers sentence-transformers faiss-cpu requests wikipedia torch`.
2. Replace `TAVILY_API_KEY` with your actual key.
3. Ensure `my_notes.md` exists with academic content.
4. Run with `python chatbot.py`.

---

## Testing and Debugging

Test your chatbot with these examples:
- **Tavily**: “Search for the latest AI breakthroughs.”
- **Wikipedia**: “Who is Alan Turing?”
- **Calculator**: “Calculate 15 * 23.”
- **RAG**: “What did my notes say about transformers?”
- **Multi-turn**: “What’s AI?” followed by “Tell me more.”

**Debugging Tips**:
- Print `select_tool` output to verify tool selection.
- Check `conversation_history` to ensure memory works.
- Handle tool failures (e.g., “Search failed”) by falling back to RAG (optional enhancement).

---

## Resources

- [Tavily Docs](https://docs.tavily.com/examples/use-cases/data-enrichment): Guide to using the search API.

---

[⬆️ Back to Top](#week-3-tools-and-memory-in-ai-agents) | [Project README](../Readme.md) | [Assignment](assignment.md)
