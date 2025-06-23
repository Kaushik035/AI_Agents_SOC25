# Week 4: Multi-turn Context-Aware Agents - Tool Chaining

In the "State Management" section, you enhanced your Study Buddy with conversation history, context optimization, entity tracking, and summarization using OpenAI’s API. Now, we’ll focus on **tool chaining**, enabling your agent to combine multiple tools in a coordinated way to handle complex queries. Tool chaining involves executing tools in sequence, validating their outputs, managing dependencies, and applying conditional logic to determine execution flow. This section covers:

- **Sequential Tool Execution Patterns**: Orchestrating multiple tools in a defined order.
- **Output Validation and Error Recovery**: Ensuring tool outputs are reliable and handling failures.
- **Dependency Management Between Tools**: Coordinating inputs and outputs across tools.
- **Conditional Execution Flows**: Deciding which tools to use based on query context.

This material uses the OpenAI API (via the provided proxy server) and integrates with the Week 3 tools (Tavily, Wikipedia, calculator) and the Week 4 state management system.

---

## Sequential Tool Execution Patterns

### What are Sequential Tool Execution Patterns?

Sequential tool execution involves running multiple tools in a specific order to fulfill a query. For example, to answer “What’s the population of Paris?”, the agent might first use Wikipedia to get the population and then a calculator to convert units if needed. This ensures each tool’s output feeds into the next step logically.

### Why is it Important?

- **Complex Queries**: Handles questions requiring multiple steps (e.g., research then computation).
- **Modularity**: Breaks tasks into manageable tool-based steps.
- **Accuracy**: Ensures each step is completed correctly before proceeding.

### How to Implement Sequential Tool Execution

1. **Define Tool Sequence**: Map query types to tool chains (e.g., “population” → Wikipedia → Calculator).
2. **Execute Sequentially**: Call each tool in order, passing outputs as inputs where needed.
3. **Integrate with LLM**: Use OpenAI to interpret or refine intermediate outputs.

### Example Implementation

Here’s a function to execute a Wikipedia search followed by text processing with OpenAI:

```python
import requests
import wikipedia

API_URL = "http://socapi.deepaksilaych.me/student1"

def call_openai(messages):
    """Call OpenAI API."""
    headers = {"Content-Type": "application/json"}
    payload = {"model": "gpt-3.5-turbo", "messages": messages}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "API error."

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No page found."

def sequential_tool_chain(query):
    """Execute Wikipedia search then refine with OpenAI."""
    if "population" in query.lower():
        # Step 1: Get raw data from Wikipedia
        wiki_result = search_wikipedia(query)
        if "No page found" in wiki_result or "Multiple options" in wiki_result:
            return wiki_result
        
        # Step 2: Refine with OpenAI
        messages = [
            {"role": "system", "content": "Extract and summarize the population information."},
            {"role": "user", "content": f"From: {wiki_result}\nExtract the population."}
        ]
        return call_openai(messages)
    return "Query not supported for this tool chain."

# Example usage
print(sequential_tool_chain("Population of Paris"))
```

**Output** (example):
```
The population of Paris is approximately 2.2 million.
```

### Notes
- **Sequence**: Wikipedia provides raw data; OpenAI extracts and summarizes.
- **Extensibility**: Add more tools (e.g., calculator for unit conversion) to the chain.
- **Setup**: Requires `wikipedia` (`pip install wikipedia`) and `requests`.

---

## Output Validation and Error Recovery

### What is Output Validation and Error Recovery?

Output validation checks if a tool’s output is valid and relevant before using it. Error recovery handles cases where a tool fails (e.g., API errors, irrelevant results) by retrying, falling back to another tool, or informing the user.

### Why is it Important?

- **Reliability**: Prevents incorrect or incomplete responses.
- **Robustness**: Ensures the agent continues functioning despite failures.

### How to Implement Output Validation and Error Recovery

1. **Validation**: Check output for expected format, content, or relevance.
2. **Error Handling**: Retry failed tools, fall back to alternatives (e.g., RAG), or return a user-friendly message.
3. **Logging**: Record errors for debugging.

### Example Implementation

Here’s a validated tool chain with error recovery:

```python
def validate_wiki_output(output):
    """Check if Wikipedia output is valid."""
    if "No page found" in output or "Multiple options" in output or len(output.split()) < 10:
        return False, output
    return True, output

def sequential_tool_chain_with_validation(query):
    """Execute tool chain with validation and error recovery."""
    if "population" in query.lower():
        # Step 1: Try Wikipedia
        wiki_result = search_wikipedia(query)
        is_valid, result = validate_wiki_output(wiki_result)
        if not is_valid:
            # Fallback to OpenAI for general knowledge
            messages = [{"role": "user", "content": query}]
            return call_openai(messages)
        
        # Step 2: Refine with OpenAI
        messages = [
            {"role": "system", "content": "Extract and summarize the population information."},
            {"role": "user", "content": f"From: {wiki_result}\nExtract the population."}
        ]
        return call_openai(messages)
    return "Query not supported."

# Example usage
print(sequential_tool_chain_with_validation("Population of Narnia"))  # Fallback case
```

**Output** (example):
```
Narnia is a fictional place, so no real population data exists.
```

### Notes
- **Validation**: Checks for short or error-prone outputs.
- **Recovery**: Falls back to OpenAI’s general knowledge if Wikipedia fails.
- **Logging**: Add `print(f"Error: {result}")` for debugging.

---

## Dependency Management Between Tools

### What is Dependency Management Between Tools?

Dependency management ensures that the output of one tool is correctly formatted and passed as input to the next tool in the chain. For example, a calculator might depend on a Wikipedia tool to provide numerical data.

### Why is it Important?

- **Correctness**: Ensures tools receive compatible inputs.
- **Efficiency**: Avoids redundant tool calls.

### How to Implement Dependency Management

1. **Define Dependencies**: Specify which tools’ outputs feed into others.
2. **Format Conversion**: Transform outputs to match the next tool’s input requirements.
3. **Error Propagation**: Handle cases where a dependency fails.

### Example Implementation

Here’s a chain where Wikipedia provides data for a calculator:

```python
import re

def extract_number(text):
    """Extract the first number from text."""
    match = re.search(r'\d+(?:,\d+)*', text)
    return match.group(0).replace(",", "") if match else None

def calculate(expression):
    """Simple calculator for arithmetic."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "Calculation error."

def tool_chain_with_dependency(query):
    """Chain Wikipedia and calculator for population conversion."""
    if "population in thousands" in query.lower():
        # Step 1: Get population from Wikipedia
        base_query = query.replace("in thousands", "").strip()
        wiki_result = search_wikipedia(base_query)
        is_valid, result = validate_wiki_output(wiki_result)
        if not is_valid:
            return result
        
        # Step 2: Extract number
        population = extract_number(wiki_result)
        if not population:
            return "No population data found."
        
        # Step 3: Convert to thousands using calculator
        calc_result = calculate(f"{population} / 1000")
        if "error" in calc_result.lower():
            return "Calculation failed."
        
        # Step 4: Format with OpenAI
        messages = [
            {"role": "system", "content": "Format the population in thousands."},
            {"role": "user", "content": f"Population: {calc_result} thousand"}
        ]
        return call_openai(messages)
    return "Query not supported."

# Example usage
print(tool_chain_with_dependency("Population of Paris in thousands"))
```

**Output** (example):
```
The population of Paris is approximately 2200 thousand.
```

### Notes
- **Dependency**: Wikipedia’s output is processed to extract a number for the calculator.
- **Safety**: `eval` is restricted for security; consider `ast` for safer parsing (Week 3).
- **Formatting**: OpenAI refines the final output for clarity.

---

## Conditional Execution Flows

### What are Conditional Execution Flows?

Conditional execution flows determine which tools or chains to run based on the query’s content or context. For example, a query about “latest news” might trigger Tavily, while “calculate” triggers the calculator.

### Why is it Important?

- **Flexibility**: Adapts to diverse query types.
- **Efficiency**: Avoids unnecessary tool calls.

### How to Implement Conditional Execution Flows

1. **Query Analysis**: Use keywords or intent detection to select tools.
2. **Decision Tree**: Define conditions for each tool or chain.
3. **Fallback**: Default to RAG or OpenAI for unhandled queries.

### Example Implementation

Here’s a conditional flow integrating all Week 3 tools and RAG:

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# RAG setup (from Week 2/3)
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

def conditional_tool_execution(query, chunks, embedder, index, tavily_api_key):
    """Execute tools based on query conditions."""
    query_lower = query.lower()
    
    if "search" in query_lower or "latest" in query_lower:
        # Tavily for web search
        url = "https://api.tavily.com/search"
        payload = {"api_key": tavily_api_key, "query": query, "search_depth": "basic", "include_answer": True}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        return "Search failed."
    
    elif "wiki" in query_lower or "who is" in query_lower:
        # Wikipedia for factual lookup
        return search_wikipedia(query)
    
    elif "calculate" in query_lower or any(op in query_lower for op in ["+", "-", "*", "/"]):
        # Calculator for math
        expression = query.split("calculate", 1)[1].strip() if "calculate" in query_lower else query
        return calculate(expression)
    
    elif "population in thousands" in query_lower:
        # Tool chain for population conversion
        return tool_chain_with_dependency(query)
    
    else:
        # RAG for note-based queries
        relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
        context = "\n".join(relevant_chunks)
        messages = [
            {"role": "system", "content": "Answer based on the notes or say 'I cannot find the answer in the provided notes.'"},
            {"role": "user", "content": f"Notes: {context}\nQuestion: {query}"}
        ]
        return call_openai(messages)
    
# Example usage
chunks = load_and_chunk_document("my_notes.md")
embeddings, embedder = generate_embeddings(chunks)
index = create_faiss_index(embeddings)
TAVILY_API_KEY = "your-tavily-api-key"
print(conditional_tool_execution("Population of Paris in thousands", chunks, embedder, index, TAVILY_API_KEY))
```

**Output** (example):
```
The population of Paris is approximately 2200 thousand.
```

### Notes
- **Conditions**: Keyword-based logic routes queries to appropriate tools or chains.
- **Fallback**: RAG handles queries not matching specific conditions.
- **Setup**: Requires `sentence-transformers`, `faiss-cpu`, `requests`, `wikipedia`.

---

## Flow Architecture

1. **Query Analysis**: Determine tool or chain based on keywords/intent.
2. **Tool Execution**: Run selected tool or chain, passing outputs as needed.
3. **Validation**: Check outputs for validity; recover if needed.
4. **Response Generation**: Use OpenAI to refine or RAG for note-based answers.
5. **Output**: Return response to user.

### Flow Diagram

```
[Query] → [Analyze Intent] → [Select Tool/Chain] → [Execute & Validate] → [Refine with OpenAI] → [Output]
```

---

## Example Integration

Here’s a complete chatbot integrating tool chaining with state management:

```python
import json
import os
import requests
import wikipedia
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

API_URL = "http://socapi.deepaksilaych.me/student1"
HISTORY_FILE = "conversation_history.json"
TAVILY_API_KEY = "your-tavily-api-key"
MAX_TOKENS = 3000

# State Management (from previous section)
conversation_history = []
entities = {}

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

# Tool Functions
def call_openai(messages):
    headers = {"Content-Type": "application/json"}
    payload = {"model": "gpt-3.5-turbo", "messages": messages}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "API error."

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple options found: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No page found."

def calculate(expression):
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "Calculation error."

def validate_wiki_output(output):
    if "No page found" in output or "Multiple options" in output or len(output.split()) < 10:
        return False, output
    return True, output

def extract_number(text):
    match = re.search(r'\d+(?:,\d+)*', text)
    return match.group(0).replace(",", "") if match else None

def tool_chain_with_dependency(query):
    if "population in thousands" in query.lower():
        base_query = query.replace("in thousands", "").strip()
        wiki_result = search_wikipedia(base_query)
        is_valid, result = validate_wiki_output(wiki_result)
        if not is_valid:
            return result
        population = extract_number(wiki_result)
        if not population:
            return "No population data found."
        calc_result = calculate(f"{population} / 1000")
        if "error" in calc_result.lower():
            return "Calculation failed."
        messages = [
            {"role": "system", "content": "Format the population in thousands."},
            {"role": "user", "content": f"Population: {calc_result} thousand"}
        ]
        return call_openai(messages)
    return "Query not supported."

# RAG Functions
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

# Main Logic
def handle_query(query, chunks, embedder, index, tavily_api_key):
    query_lower = query.lower()
    if "search" in query_lower or "latest" in query_lower:
        url = "https://api.tavily.com/search"
        payload = {"api_key": tavily_api_key, "query": query, "search_depth": "basic", "include_answer": True}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        return "Search failed."
    
    elif "wiki" in query_lower or "who is" in query_lower:
        return search_wikipedia(query)
    
    elif "calculate" in query_lower or any(op in query_lower for op in ["+", "-", "*", "/"]):
        expression = query.split("calculate", 1)[1].strip() if "calculate" in query_lower else query
        return calculate(expression)
    
    elif "population in thousands" in query_lower:
        return tool_chain_with_dependency(query)
    
    else:
        relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
        context = "\n".join(relevant_chunks)
        optimized_context = get_optimized_context(query)
        entity_context = get_entity_context(query.split()[0])
        if entity_context:
            optimized_context.append({"role": "system", "content": f"Note: User mentioned {query.split()[0]}: {entity_context}"})
        if len(conversation_history) > 10:
            summary = summarize_context()
            optimized_context.append({"role": "system", "content": f"Summary: {summary}"})
        messages = [
            {"role": "system", "content": "Answer based on the notes or say 'I cannot find the answer in the provided notes.'"},
            *[{"role": msg["role"], "content": msg["content"]} for msg in optimized_context],
            {"role": "user", "content": f"Notes: {context}\nQuestion: {query}"}
        ]
        return call_openai(messages)

def run_chatbot(file_path, tavily_api_key):
    load_history()
    chunks = load_and_chunk_document(file_path)
    embeddings, embedder = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)
    print("Welcome to the Study Buddy! Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        add_to_history("user", query)
        response = handle_query(query, chunks, embedder, index, tavily_api_key)
        add_to_history("assistant", response)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    run_chatbot("my_notes.md", TAVILY_API_KEY)
```

### Test Cases
- **Search**: “Search for latest AI news” → Tavily response.
- **Wikipedia**: “Who is Alan Turing?” → Wikipedia summary.
- **Calculator**: “Calculate 15 * 23” → “345”.
- **Tool Chain**: “Population of Paris in thousands” → “2200 thousand”.
- **RAG**: “What’s in my notes about transformers?” → Note-based answer.

