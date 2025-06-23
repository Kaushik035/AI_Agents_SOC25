# Week 4: Multi-turn Context-Aware Agents - Tools Engineering

In the "Tool Chaining" section, your Study Buddy gained the ability to orchestrate multiple tools, validate outputs, manage dependencies, and execute conditional logic, making it versatile for complex queries. Now, we’ll focus on **persona engineering**, shaping the agent’s personality, tone, and behavior to align with its role as a friendly, knowledgeable study partner. This section covers four key aspects:

- **System Prompt Engineering**: Crafting prompts to define the agent’s role and behavior.
- **Persona Engineering Design**: Tailoring responses to match desired tone and style.
- **Prompt Engineering-Specific Language Modeling**: Adapting language to specific academic domains.
- **Ethical Guardrails Implementation**: Ensuring responses are safe, respectful, and unbiased.

This material uses OpenAI’s API (via the provided proxy server) and integrates with the Week 4 state management and tool chaining systems, ensuring a cohesive learning experience for your Study Buddy.

---

## System Prompt Engineering

### What is System Prompt Design?

System prompt design involves creating initial instructions that define the agent’s role, goals, and behavior throughout the conversation. For Study Buddy, the prompt establishes it as a supportive, student-friendly tutor.

### Why is it Important?

- **Role Clarity**: Ensures the agent consistently acts as a study buddy (e.g., helpful, not authoritative).
- **Behavior Guidance**: Sets expectations for how the agent are responds to queries (e.g., explaining concepts clearly).
- **Consistency**: Maintains a unified persona across interactions.

### How to Implement System Prompt Design

1. **Define Role**: Specify the agent is (e.g., a peer tutor).
2. **Set Tone**: Instruct a friendly, clear, and concise tone.
3. **Include Goals**: Emphasize educational support and accuracy.
4. **Test and Refine**: Experiment with prompts to optimize responses.

### Example Implementation

Here’s a system prompt for Study Buddy’s role:

```python
SYSTEM_PROMPT = """
You are Study Buddy, a friendly and knowledgeable peer tutor. Your goal is to help students understand academic concepts in a clear, concise, and engaging way. Always respond with a supportive tone, breaking down complex ideas into simple explanations. Use examples when helpful and avoid jargon unless explained. If unsure, admit it and suggest a way to find the answer. Prioritize accuracy and clarity.
"""

def call_openai(query, context):
    """Call OpenAI API with system prompt."""
    headers = {"Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *[{"role": msg["role"], "content": msg["content"]} for msg in context],
        {"role": "user", "content": query}
    ]
    payload = {"model": "gpt-3.5-turbo", "messages": messages}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "API error."

# Example usage
context = []  # Empty context for simplicity
response = call_openai("What is a neural network?", context)
print(response)
```

**Output** (example):
```
A neural network is like a digital brain that helps computers learn patterns from data. Imagine teaching a friend to recognize cats by showing them pictures: they notice features like whiskers or ears. A neural network does this by layering “neurons” that process information, adjusting connections to improve predictions. For example, it might learn to identify cats in new images after training on many cat pictures. Want me to dive deeper into how it works?
```

### Notes
- **Prompt Clarity**: The prompt explicitly defines role, tone, and goals.
- **Flexibility**: Adjust `SYSTEM_PROMPT` for different subjects or styles.
- **Setup**: Requires `requests` and the API URL (`http://socapi.deepaksilaych.me/student1`).

---

## Response Style Conditioning

### What is Response Style Conditioning?

Response style conditioning shapes the agent’s tone, vocabulary, and structure to match the desired persona. For Study Buddy, this means adopting a friendly, conversational style that feels like chatting with a classmate.

### Why is it Important?

- **Engagement**: A relatable tone keeps students interested.
- **Accessibility**: Matches the user’s language level (e.g., high school vs. college).
- **Persona Consistency**: Reinforces the agent’s identity.

### How to Implement Response Style Conditioning

1. **Prompt Instructions**: Include style guidelines in the system prompt.
2. **Example Responses**: Provide few-shot examples to guide the LLM.
3. **Dynamic Adjustment**: Modify style based on user input (e.g., formal for professors).

### Example Implementation

Here’s a function to condition Study Buddy’s style:

```python
def get_style_conditioned_prompt(query, user_level="high_school"):
    """Generate prompt with style conditioning."""
    style_guidelines = {
        "high_school": "Use simple words and a casual, friendly tone like you're explaining to a friend. Include fun analogies.",
        "college": "Use precise terms and a professional yet approachable tone. Provide detailed examples."
    }
    style = style_guidelines.get(user_level, style_guidelines["high_school"])
    
    return f"""
    {SYSTEM_PROMPT}
    Additional instructions: {style}
    Respond to: {query}
    """

def handle_styled_query(query, context, user_level="high_school"):
    """Process query with style conditioning."""
    styled_prompt = get_style_conditioned_prompt(query, user_level)
    messages = [
        {"role": "system", "content": styled_prompt},
        *[{"role": msg["role"], "content": msg["content"]} for msg in context]
    ]
    return call_openai(query, context)  # Context includes history

# Example usage
context = [{"role": "user", "content": "What is calculus?"}, {"role": "assistant", "content": "Calculus is about studying change, like how fast things move or grow."}]
print(handle_styled_query("Explain derivatives", context, "high_school"))
print(handle_styled_query("Explain derivatives", context, "college"))
```

**Output** (example):
- **High School**:
```
A derivative is like checking how fast your bike is going at one exact moment. Imagine you’re riding up a hill, and your speed changes. The derivative tells you the slope of that hill at any point, showing how steep it is right there. For example, if you’re graphing distance vs. time, the derivative gives your speed!
```
- **College**:
```
A derivative represents the instantaneous rate of change of a function at a given point. Mathematically, for a function \( f(x) \), the derivative \( f'(x) \) is defined as \( \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \). For instance, if \( f(x) = x^2 \), then \( f'(x) = 2x \), indicating the slope of the tangent line at \( x \). This is fundamental in optimization and physics.
```

### Notes
- **Dynamic Style**: Adjusts based on `user_level`; extend to detect user age or expertise.
- **Examples**: Few-shot examples in the prompt can further refine style.
- **Integration**: Works with state management context.

---

## Domain-Specific Language Modeling

### What is Domain-Specific Language Modeling?

Domain-specific language modeling adapts the agent’s vocabulary and knowledge to a particular academic field (e.g., computer science, biology). For Study Buddy, this ensures accurate and relevant terminology when answering subject-specific queries.

### Why is it Important?

- **Accuracy**: Uses correct terms (e.g., “algorithm” in CS vs. “photosynthesis” in biology).
- **Relevance**: Aligns responses with the user’s study focus.
- **Credibility**: Builds trust by sounding knowledgeable.

### How to Implement Domain-Specific Language Modeling

1. **Domain Detection**: Identify the subject from the query or context.
2. **Custom Prompts**: Use domain-specific prompts with relevant terminology.
3. **Knowledge Injection**: Include domain facts or examples in the prompt.

### Example Implementation

Here’s a function to adapt language for different domains:

```python
DOMAIN_PROMPTS = {
    "computer_science": """
    You are an expert in computer science. Use terms like 'algorithm', 'data structure', or 'runtime complexity' accurately. Explain concepts with coding examples when relevant. Avoid oversimplifying technical details.
    """,
    "biology": """
    You are a biology tutor. Use terms like 'cell', 'ecosystem', or 'DNA' accurately. Include biological processes and examples, like how enzymes work or species interactions.
    """,
    "default": SYSTEM_PROMPT
}

def detect_domain(query):
    """Simple keyword-based domain detection."""
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["code", "algorithm", "python", "database"]):
        return "computer_science"
    elif any(kw in query_lower for kw in ["cell", "dna", "evolution", "plant"]):
        return "biology"
    return "default"

def handle_domain_specific_query(query, context):
    """Process query with domain-specific language."""
    domain = detect_domain(query)
    domain_prompt = DOMAIN_PROMPTS[domain]
    messages = [
        {"role": "system", "content": domain_prompt},
        *[{"role": msg["role"], "content": msg["content"]} for msg in context],
        {"role": "user", "content": query}
    ]
    return call_openai(messages)

# Example usage
context = [{"role": "user", "content": "What is a function?"}, {"role": "assistant", "content": "Depends on the subject!"}]
print(handle_domain_specific_query("What is a function?", context))  # CS context
print(handle_domain_specific_query("What is a cell?", context))    # Biology context
```

**Output** (example):
- **CS**:
```
In computer science, a function is a block reusable block of code that performs a specific task. For example, in Python, you might write:
```python
def add(a, b):
    return a + b
```
This function takes two numbers and returns their sum. Functions help organize code and improve modularity.
```
- **Biology**:
```
A cell is the basic building block of life, the smallest unit that can perform all life processes. For example, a plant cell contains chloroplasts, which use sunlight to produce energy through a process called photosynthesis. Cells work together to form tissues and organisms.
```

### Notes
- **Detection**: Keyword-based detection is simple; improve with embeddings (e.g., `sentence-transformers`).
- **Extensibility**: Add more domains (e.g., physics, history) to `DOMAIN_PROMPTS`.
- **Tool Integration**: Combine with Wikipedia or RAG for domain-specific facts.

---

## Ethical Guardrails Implementation

### What is Ethical Guardrails Implementation?

An implementation of ethical guardrails ensures the agent’s responses are safe, inclusive, and free from biases or harmful content. For Study Buddy, this means avoiding offensive language, misinformation, or biased assumptions while promoting respectful interactions.

### Why is it Important?

- **Safety**: Protects users from harmful or inappropriate content.
- **Inclusivity**: Ensures responses respect diverse backgrounds.
- **Trust**: Builds confidence in the agent’s reliability.

### How to Implement Ethical Guardrails

1. **Prompt Constraints**: Instruct the LLM to avoid harmful content.
2. **Response Filtering**: Check outputs for sensitive terms or biases.
3. **Bias Detection**: Use simple rules or external tools to flag issues.
4. **User Feedback**: Allow users to report problematic responses.

### Example Implementation

Here’s a function to enforce ethical guardrails:

```python
SENSITIVE_TERMS = ["hate", "violence", "discriminate", "offensive"]  # Extend as needed

def check_ethical_compliance(text):
    """Check response for ethical issues."""
    text_lower = text.lower()
    for term in SENSITIVE_TERMS:
        if term in text_lower:
            return False, f"Response contains sensitive term: {term}"
    # Simple bias check (extend with NLP tools)
    if any(phrase in text_lower for phrase in ["better than", "superior race", "inferior"]):
        return False, "Potential bias detected."
    return True, "Compliant"

def handle_query_with_guardrails(query, context):
    """Process query with ethical guardrails."""
    # Generate response
    response = handle_domain_specific_query(query, context)
    
    # Check compliance
    is_compliant, message = check_ethical_compliance(response)
    if not is_compliant:
        return f"Sorry, I can’t provide that response due to ethical concerns ({message}). Please rephrase your question."
    
    return response

# Example usage
context = [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is artificial intelligence."}]
print(handle_query_with_guardrails("Why is one group better than another?", context))  # Triggers guardrail
print(handle_query_with_guardrails("What is a neural network?", context))            # Normal response
```

**Output** (example):
- **Biased Query**:
```
Sorry, I can’t provide that response due to ethical concerns (Potential bias detected.). Please rephrase your question.
```
- **Normal Query**:
```
A neural network is like a digital brain that learns patterns from data. It’s made of layers of “neurons” that process information, like recognizing cats in photos after training on many images. Want a deeper dive?
```

### Notes
- **Filtering**: `SENSITIVE_TERMS` is basic; use libraries like `detoxify` for advanced moderation.
- **Prompt**: Add “Avoid biased or harmful content” to `SYSTEM_PROMPT` for stronger guardrails.
- **Feedback**: Log flagged responses to improve the system.

---

## Flow Architecture

1. **Query Input**: Receive user query and update conversation history.
2. **Domain Detection**: Identify the academic subject.
3. **Style Conditioning**: Apply user-level tone (e.g., high school vs. college).
4. **Prompt Assembly**: Combine system prompt, domain prompt, and context.
5. **Response Generation**: Call OpenAI with the crafted prompt.
6. **Ethical Check**: Validate response for safety and compliance.
7. **Output**: Return response or ethical warning.

### Flow Diagram

```
[Query] → [Detect Domain] → [Apply Style] → [Assemble Prompt] → [Call OpenAI] → [Check Ethics] → [Output]
```

---

## Example Integration

Here’s a complete chatbot integrating persona engineering with prior systems:

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

# Persona Engineering
SYSTEM_PROMPT = """
You are Study Buddy, a friendly and knowledgeable peer tutor. Your goal is to help students understand academic concepts in a clear, concise, and engaging way. Always respond with a supportive tone, breaking down complex ideas into simple explanations. Use examples when helpful and avoid jargon unless explained. If unsure, admit it and suggest a way to find the answer. Prioritize accuracy, clarity, and avoid biased or harmful content.
"""

SENSITIVE_TERMS = ["hate", "violence", "discriminate", "offensive"]

DOMAIN_PROMPTS = {
    "computer_science": """
    You are an expert in computer science. Use terms like 'algorithm', 'data structure', or 'runtime complexity' accurately. Explain concepts with coding examples when relevant. Avoid oversimplifying technical details.
    """,
    "biology": """
    You are a biology tutor. Use terms like 'cell', 'ecosystem', or 'DNA' accurately. Include biological processes and examples, like how enzymes work or species interactions.
    """,
    "default": SYSTEM_PROMPT
}

def detect_domain(query):
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["code", "algorithm", "python", "database"]):
        return "computer_science"
    elif any(kw in query_lower for kw in ["cell", "dna", "evolution", "plant"]):
        return "biology"
    return "default"

def get_style_conditioned_prompt(query, user_level="high_school"):
    style_guidelines = {
        "high_school": "Use simple words and a casual, friendly tone like you're explaining to a friend. Include fun analogies.",
        "college": "Use precise terms and a professional yet approachable tone. Provide detailed examples."
    }
    style = style_guidelines.get(user_level, style_guidelines["high_school"])
    domain = detect_domain(query)
    domain_prompt = DOMAIN_PROMPTS[domain]
    return f"""
    {domain_prompt}
    Additional instructions: {style}
    Respond to: {query}
    """

def check_ethical_compliance(text):
    text_lower = text.lower()
    for term in SENSITIVE_TERMS:
        if term in text_lower:
            return False, f"Response contains sensitive term: {term}"
    if any(phrase in text_lower for phrase in ["better than", "superior race", "inferior"]):
        return False, "Potential bias detected."
    return True, "Compliant"

# State Management
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
    return call_openai(summary_context)

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
def handle_query(query, chunks, embedder, index, tavily_api_key, user_level="high_school"):
    query_lower = query.lower()
    context = get_optimized_context(query)
    extract_entities(query)
    
    # Tool-based queries
    if "search" in query_lower or "latest" in query_lower:
        url = "https://api.tavily.com/search"
        payload = {"api_key": tavily_api_key, "query": query, "search_depth": "basic", "include_answer": True}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json().get("answer", "No answer found.")
        else:
            result = "Search failed."
    elif "wiki" in query_lower or "who is" in query_lower:
        result = search_wikipedia(query)
    elif "calculate" in query_lower or any(op in query_lower for op in ["+", "-", "*", "/"]):
        expression = query.split("calculate", 1)[1].strip() if "calculate" in query_lower else query
        result = calculate(expression)
    elif "population in thousands" in query_lower:
        result = tool_chain_with_dependency(query)
    else:
        # RAG or general query with persona
        relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
        note_context = "\n".join(relevant_chunks)
        entity_context = get_entity_context(query.split()[0])
        if entity_context:
            context.append({"role": "system", "content": f"Note: User mentioned {query.split()[0]}: {entity_context}"})
        if len(conversation_history) > 10:
            summary = summarize_context()
            context.append({"role": "system", "content": f"Summary: {summary}"})
        styled_prompt = get_style_conditioned_prompt(query, user_level)
        messages = [
            {"role": "system", "content": styled_prompt},
            *[{"role": msg["role"], "content": msg["content"]} for msg in context],
            {"role": "user", "content": f"Notes: {note_context}\nQuestion: {query}"}
        ]
        result = call_openai(messages)
    
    # Ethical check
    is_compliant, message = check_ethical_compliance(result)
    if not is_compliant:
        return f"Sorry, I can’t provide that response due to ethical concerns ({message}). Please rephrase your question."
    
    return result

def run_chatbot(file_path, tavily_api_key, user_level="high_school"):
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
        response = handle_query(query, chunks, embedder, index, tavily_api_key, user_level)
        add_to_history("assistant", response)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    run_chatbot("my_notes.md", TAVILY_API_KEY)
```

### Test Cases
- **General**: “What is a neural network?” → Friendly, analogy-rich response (high school).
- **Domain-Specific**: “What is a function in Python?” → CS-specific with code example.
- **Style**: “Explain derivatives” → Casual (high school) vs. technical (college).
- **Ethical**: “Why is one group better than another?” → Blocked with warning.
- **Tool Chain**: “Population of Paris in thousands” → Tool-chained response.
