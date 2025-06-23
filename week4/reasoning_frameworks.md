
# Week 4: Multi-turn Context-Aware Agents - Reasoning Frameworks

In the "Persona Engineering" section, your Study Buddy gained a well-defined, adaptable, and ethical persona, delivering tailored responses as a friendly peer tutor. Now, we’ll focus on **reasoning frameworks**, equipping the agent with structured thinking processes to handle complex queries systematically and reliably. Reasoning frameworks enable the agent to plan its approach, execute actions, refine results, and make confident decisions. This section covers four key components:

- **Plan-Execute-Refine Pattern**: Structuring query resolution into planning, execution, and refinement steps.
- **Self-Correction Mechanisms**: Detecting and fixing errors in responses.
- **Fallback Strategies**: Handling cases where primary approaches fail.
- **Confidence-Based Decision Making**: Evaluating and selecting responses based on confidence levels.

This material uses OpenAI’s API (via the provided proxy server) and integrates with the Week 4 state management, tool chaining, and persona engineering systems, ensuring a cohesive learning experience for your Study Buddy.

---

## Plan-Execute-Refine Pattern

### What is the Plan-Execute-Refine Pattern?

The Plan-Execute-Refine pattern is a structured approach where the agent first plans how to answer a query, executes the plan using tools or knowledge, and refines the output for accuracy and clarity. For Study Buddy, this ensures systematic handling of complex queries like “Solve a quadratic equation step-by-step.”

### Why is it Important?

- **Structure**: Breaks down complex tasks into manageable steps.
- **Clarity**: Produces well-organized, step-by-step responses.
- **Reliability**: Reduces errors by reviewing and refining outputs.

### How to Implement Plan-Execute-Refine

1. **Plan**: Use OpenAI to outline steps needed to answer the query.
2. **Execute**: Perform each step, using tools (e.g., calculator) or LLM knowledge.
3. **Refine**: Review the output for errors or improvements, adjusting as needed.

### Example Implementation

Here’s a function implementing the pattern for a math query:

```python
import requests

API_URL = "http://socapi.deepaksilaych.me/student1"

def call_openai(messages):
    """Call OpenAI API."""
    headers = {"Content-Type": "application/json"}
    payload = {"model": "gpt-3.5-turbo", "messages": messages}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "API error."

def plan_execute_refine(query, context):
    """Implement Plan-Execute-Refine pattern."""
    if "quadratic equation" in query.lower():
        # Step 1: Plan
        plan_prompt = f"""
        You are Study Buddy, a peer tutor. Outline a step-by-step plan to solve: {query}.
        Return the plan as a numbered list.
        """
        plan_messages = [{"role": "system", "content": plan_prompt}, *context]
        plan = call_openai(plan_messages)
        
        # Step 2: Execute
        execute_prompt = f"""
        Follow this plan to solve: {query}
        Plan:
        {plan}
        Provide a detailed solution, showing all work.
        """
        execute_messages = [{"role": "system", "content": execute_prompt}, *context]
        execution = call_openai(execute_messages)
        
        # Step 3: Refine
        refine_prompt = f"""
        Review this solution for accuracy and clarity:
        {execution}
        Correct any errors and improve clarity. Return the final solution.
        """
        refine_messages = [{"role": "system", "content": refine_prompt}, *context]
        refined_solution = call_openai(refine_messages)
        
        return f"Plan:\n{plan}\n\nSolution:\n{refined_solution}"
    
    return "Query not supported for this pattern."

# Example usage
context = []  # Empty for simplicity
print(plan_execute_refine("Solve the quadratic equation x^2 - 5x + 6 = 0", context))
```

**Output** (example):
```
Plan:
1. Identify the coefficients a, b, and c from the equation ax^2 + bx + c = 0.
2. Apply the quadratic formula x = [-b ± sqrt(b^2 - 4ac)] / (2a).
3. Calculate the discriminant (b^2 - 4ac) to check the nature of roots.
4. Compute the two possible values of x.
5. Verify the solutions by substituting back into the original equation.

Solution:
Given the equation x^2 - 5x + 6 = 0, we identify a = 1, b = -5, c = 6.

Using the quadratic formula x = [-b ± sqrt(b^2 - 4ac)] / (2a):
- Discriminant = (-5)^2 - 4(1)(6) = 25 - 24 = 1.
- x = [5 ± sqrt(1)] / (2*1) = [5 ± 1] / 2.
- x1 = (5 + 1) / 2 = 6 / 2 = 3.
- x2 = (5 - 1) / 2 = 4 / 2 = 2.

Verification:
- For x = 3: 3^2 - 5(3) + 6 = 9 - 15 + 6 = 0 (correct).
- For x = 2: 2^2 - 5(2) + 6 = 4 - 10 + 6 = 0 (correct).

The solutions are x = 2 and x = 3.
```

### Notes
- **Prompt Design**: Each step uses clear instructions to guide OpenAI.
- **Extensibility**: Adapt for other query types (e.g., “Explain photosynthesis”).
- **Context**: Integrates with conversation history for coherence.

---

## Self-Correction Mechanisms

### What are Self-Correction Mechanisms?

Self-correction mechanisms enable the agent to detect and fix errors in its responses, such as incorrect calculations or factual inaccuracies. For Study Buddy, this ensures reliable educational content.

### Why is it Important?

- **Accuracy**: Corrects mistakes before presenting to the user.
- **Trust**: Enhances user confidence in the agent’s responses.
- **Learning Support**: Provides accurate guidance for students.

### How to Implement Self-Correction Mechanisms

1. **Error Detection**: Use OpenAI to review responses for inconsistencies.
2. **Correction**: Generate a corrected response if errors are found.
3. **Validation**: Re-check the corrected output.

### Example Implementation

Here’s a function for self-correction:

```python
def self_correct_response(query, initial_response, context):
    """Detect and correct errors in a response."""
    # Step 1: Detect errors
    detect_prompt = f"""
    You are Study Buddy, a peer tutor. Review this response for errors (factual, logical, or computational):
    Query: {query}
    Response: {initial_response}
    If errors are found, explain them. If none, say "No errors detected."
    """
    detect_messages = [{"role": "system", "content": detect_prompt}, *context]
    error_report = call_openai(detect_messages)
    
    if "no errors detected" in error_report.lower():
        return initial_response
    
    # Step 2: Correct errors
    correct_prompt = f"""
    Based on this error report, provide a corrected response to the query:
    Query: {query}
    Original Response: {initial_response}
    Error Report: {error_report}
    """
    correct_messages = [{"role": "system", "content": correct_prompt}, *context]
    corrected_response = call_openai(correct_messages)
    
    return corrected_response

def handle_query_with_correction(query, context):
    """Generate and correct a response."""
    initial_prompt = f"""
    You are Study Buddy, a peer tutor. Answer clearly and accurately: {query}
    """
    initial_messages = [{"role": "system", "content": initial_prompt}, *context]
    initial_response = call_openai(initial_messages)
    return self_correct_response(query, initial_response, context)

# Example usage
context = []
print(handle_query_with_correction("What is 15 * 23?", context))
```

**Output** (example):
- If initial response is incorrect (e.g., “15 * 23 = 325”):
```
The correct answer is 345. The original response had a computational error: 15 * 23 = 15 * (20 + 3) = 15 * 20 + 15 * 3 = 300 + 45 = 345.
```
- If correct:
```
15 * 23 = 345
```

### Notes
- **Detection**: Relies on OpenAI’s ability to spot errors; improve with external validators (e.g., calculator).
- **Recursion**: Limit correction iterations to avoid infinite loops.
- **Integration**: Works with persona engineering prompts for consistent tone.

---

## Fallback Strategies

### What are Fallback Strategies?

Fallback strategies provide alternative approaches when the primary method fails (e.g., tool errors, unclear queries). For Study Buddy, this ensures the agent can still respond helpfully, even in challenging cases.

### Why is it Important?

- **Robustness**: Prevents the agent from failing silently.
- **User Experience**: Maintains engagement with partial or alternative answers.
- **Flexibility**: Adapts to unexpected inputs.

### How to Implement Fallback Strategies

1. **Primary Attempt**: Try the planned approach (e.g., tool chain or RAG).
2. **Fallback Options**: Use alternatives like general LLM knowledge or rephrasing.
3. **User Communication**: Inform the user if the query can’t be fully answered.

### Example Implementation

Here’s a function with fallback strategies:

```python
def handle_query_with_fallback(query, chunks, embedder, index, tavily_api_key, context):
    """Process query with fallback strategies."""
    query_lower = query.lower()
    
    # Primary: Tool-based or RAG
    try:
        if "calculate" in query_lower:
            expression = query.split("calculate", 1)[1].strip()
            result = calculate(expression)
            if "error" not in result.lower():
                return result
        elif "search" in query_lower:
            url = "https://api.tavily.com/search"
            payload = {"api_key": tavily_api_key, "query": query, "search_depth": "basic"}
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get("answer", "No answer found.")
        else:
            relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
            note_context = "\n".join(relevant_chunks)
            messages = [
                {"role": "system", "content": "Answer based on notes or general knowledge."},
                *[{"role": msg["role"], "content": msg["content"]} for msg in context],
                {"role": "user", "content": f"Notes: {note_context}\nQuestion: {query}"}
            ]
            result = call_openai(messages)
            if "cannot find" not in result.lower():
                return result
    except Exception as e:
        pass  # Proceed to fallback
    
    # Fallback: General LLM knowledge
    fallback_prompt = f"""
    You are Study Buddy, a peer tutor. The primary method failed. Answer {query} using general knowledge or suggest an alternative approach.
    """
    fallback_messages = [{"role": "system", "content": fallback_prompt}, *context]
    return call_openai(fallback_messages)

# Example usage
context = []
chunks = ["Sample note content"]
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(handle_query_with_fallback("Calculate invalid", chunks, embedder, index, "your-tavily-api-key", context))
```

**Output** (example):
```
Sorry, I couldn’t calculate that due to an invalid expression. Could you clarify the calculation, like “calculate 5 + 3”? Alternatively, I can explain a math concept if you’d like!
```

### Notes
- **Fallback**: Defaults to OpenAI’s knowledge if tools or RAG fail.
- **User-Friendly**: Informs user of issues and offers alternatives.
- **Setup**: Requires `sentence-transformers`, `faiss-cpu`, `requests`.

---

## Confidence-Based Decision Making

### What is Confidence-Based Decision Making?

Confidence-based decision making involves evaluating multiple response options and selecting the one with the highest confidence score, based on criteria like coherence or tool reliability. For Study Buddy, this ensures the most reliable answer is presented.

### Why is it Important?

- **Quality**: Chooses the best response among alternatives.
- **Transparency**: Can inform users of confidence levels for trust.
- **Adaptability**: Balances tool outputs and LLM knowledge.

### How to Implement Confidence-Based Decision Making

1. **Generate Options**: Produce multiple responses (e.g., tool vs. LLM).
2. **Score Confidence**: Use heuristics (e.g., length, keyword match) or OpenAI to evaluate.
3. **Select Best**: Choose the highest-scored response.

### Example Implementation

Here’s a function for confidence-based selection:

```python
def evaluate_confidence(response, query):
    """Score response confidence based on heuristics."""
    score = 0
    # Heuristic 1: Length (longer responses often more detailed)
    score += min(len(response.split()) / 50, 0.4)  # Cap at 0.4
    # Heuristic 2: Keyword overlap
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    overlap = len(query_words & response_words) / len(query_words)
    score += overlap * 0.4
    # Heuristic 3: No error indicators
    if "error" not in response.lower() and "cannot find" not in response.lower():
        score += 0.2
    return min(score, 1.0)  # Normalize to [0, 1]

def handle_query_with_confidence(query, chunks, embedder, index, tavily_api_key, context):
    """Select best response based on confidence."""
    responses = []
    
    # Option 1: RAG
    relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
    note_context = "\n".join(relevant_chunks)
    rag_messages = [
        {"role": "system", "content": "Answer based on notes."},
        *[{"role": msg["role"], "content": msg["content"]} for msg in context],
        {"role": "user", "content": f"Notes: {note_context}\nQuestion: {query}"}
    ]
    rag_response = call_openai(rag_messages)
    responses.append(("RAG", rag_response, evaluate_confidence(rag_response, query)))
    
    # Option 2: Tool (Wikipedia for factual queries)
    if "who is" in query.lower():
        wiki_response = search_wikipedia(query)
        if validate_wiki_output(wiki_response)[0]:
            responses.append(("Wikipedia", wiki_response, evaluate_confidence(wiki_response, query)))
    
    # Option 3: General LLM
    llm_prompt = f"You are Study Buddy, a peer tutor. Answer: {query}"
    llm_messages = [{"role": "system", "content": llm_prompt}, *context]
    llm_response = call_openai(llm_messages)
    responses.append(("LLM", llm_response, evaluate_confidence(llm_response, query)))
    
    # Select best
    best_source, best_response, best_score = max(responses, key=lambda x: x[2])
    return f"Response (from {best_source}, confidence {best_score:.2f}): {best_response}"

# Example usage
context = []
print(handle_query_with_confidence("Who is Alan Turing?", chunks, embedder, index, "your-tavily-api-key", context))
```

**Output** (example):
```
Response (from Wikipedia, confidence 0.85): Alan Mathison Turing OBE FRS (/ˈtjʊərɪŋ/; 23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist. Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer.
```

### Notes
- **Scoring**: Heuristics are simple; improve with OpenAI-based evaluation prompts.
- **Multiple Options**: Balances tools, RAG, and LLM for robustness.
- **Transparency**: Reports confidence for user trust.

---

## Flow Architecture

1. **Query Input**: Receive query and update conversation history.
2. **Planning**: Outline steps for complex queries (Plan-Execute-Refine).
3. **Execution**: Run tools, RAG, or LLM based on query type.
4. **Self-Correction**: Review and fix errors in the response.
5. **Confidence Evaluation**: Score multiple response options.
6. **Fallback**: Use alternative methods if primary fails.
7. **Output**: Return the best response with ethical checks.

### Flow Diagram

```
[Query] → [Plan] → [Execute] → [Self-Correct] → [Evaluate Confidence] → [Fallback if Needed] → [Ethical Check] → [Output]
```

---

## Example Integration

Here’s a complete chatbot integrating reasoning frameworks with prior systems:

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
    return f"""
    {DOMAIN_PROMPTS[domain]}
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

# Reasoning Frameworks
def plan_execute_refine(query, context):
    if "quadratic equation" in query.lower():
        plan_prompt = f"""
        You are Study Buddy, a peer tutor. Outline a step-by-step plan to solve: {query}.
        Return the plan as a numbered list.
        """
        plan_messages = [{"role": "system", "content": plan_prompt}, *[{"role": msg["role"], "content": msg["content"]} for msg in context]]
        plan = call_openai(plan_messages)
        
        execute_prompt = f"""
        Follow this plan to solve: {query}
        Plan:
        {plan}
        Provide a detailed solution, showing all work.
        """
        execute_messages = [{"role": "system", "content": execute_prompt}, *[{"role": msg["role"], "content": msg["content"]} for msg in context]]
        execution = call_openai(execute_messages)
        
        refine_prompt = f"""
        Review this solution for accuracy and clarity:
        {execution}
        Correct any errors and improve clarity. Return the final solution.
        """
        refine_messages = [{"role": "system", "content": refine_prompt}, *[{"role": msg["role"], "content": msg["content"]} for msg in context]]
        refined_solution = call_openai(refine_messages)
        
        return f"Plan:\n{plan}\n\nSolution:\n{refined_solution}"
    return None

def self_correct_response(query, initial_response, context):
    detect_prompt = f"""
    You are Study Buddy, a peer tutor. Review this response for errors (factual, logical, or computational):
    Query: {query}
    Response: {initial_response}
    If errors are found, explain them. If none, say "No errors detected."
    """
    detect_messages = [{"role": "system", "content": detect_prompt}, *[{"role": msg["role"], "content": msg["content"]} for msg in context]]
    error_report = call_openai(detect_messages)
    
    if "no errors detected" in error_report.lower():
        return initial_response
    
    correct_prompt = f"""
    Based on this error report, provide a corrected response to the query:
    Query: {query}
    Original Response: {initial_response}
    Error Report: {error_report}
    """
    correct_messages = [{"role": "system", "content": correct_prompt}, *[{"role": msg["role"], "content": msg["content"]} for msg in context]]
    return call_openai(correct_messages)

def evaluate_confidence(response, query):
    score = 0
    score += min(len(response.split()) / 50, 0.4)
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    overlap = len(query_words & response_words) / len(query_words)
    score += overlap * 0.4
    if "error" not in response.lower() and "cannot find" not in response.lower():
        score += 0.2
    return min(score, 1.0)

def handle_query_with_reasoning(query, chunks, embedder, index, tavily_api_key, context, user_level="high_school"):
    query_lower = query.lower()
    
    # Plan-Execute-Refine for specific queries
    if "quadratic equation" in query_lower:
        result = plan_execute_refine(query, context)
        if result:
            return self_correct_response(query, result, context)
    
    # Generate multiple response options
    responses = []
    
    # Option 1: RAG
    try:
        relevant_chunks = retrieve_chunks(query, embedder, index, chunks)
        note_context = "\n".join(relevant_chunks)
        rag_messages = [
            {"role": "system", "content": get_style_conditioned_prompt(query, user_level)},
            *[{"role": msg["role"], "content": msg["content"]} for msg in context],
            {"role": "user", "content": f"Notes: {note_context}\nQuestion: {query}"}
        ]
        rag_response = call_openai(rag_messages)
        responses.append(("RAG", rag_response, evaluate_confidence(rag_response, query)))
    except Exception:
        pass
    
    # Option 2: Tool
    try:
        if "search" in query_lower:
            url = "https://api.tavily.com/search"
            payload = {"api_key": tavily_api_key, "query": query, "search_depth": "basic"}
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                tool_response = response.json().get("answer", "No answer found.")
                responses.append(("Tavily", tool_response, evaluate_confidence(tool_response, query)))
        elif "who is" in query_lower:
            wiki_response = search_wikipedia(query)
            if validate_wiki_output(wiki_response)[0]:
                responses.append(("Wikipedia", wiki_response, evaluate_confidence(wiki_response, query)))
        elif "calculate" in query_lower:
            expression = query.split("calculate", 1)[1].strip()
            calc_response = calculate(expression)
            if "error" not in calc_response.lower():
                responses.append(("Calculator", calc_response, evaluate_confidence(calc_response, query)))
        elif "population in thousands" in query_lower:
            tool_response = tool_chain_with_dependency(query)
            responses.append(("Tool Chain", tool_response, evaluate_confidence(tool_response, query)))
    except Exception:
        pass
    
    # Option 3: General LLM
    llm_messages = [
        {"role": "system", "content": get_style_conditioned_prompt(query, user_level)},
        *[{"role": msg["role"], "content": msg["content"]} for msg in context],
        {"role": "user", "content": query}
    ]
    llm_response = call_openai(llm_messages)
    responses.append(("LLM", llm_response, evaluate_confidence(llm_response, query)))
    
    # Select best response
    if responses:
        best_source, best_response, best_score = max(responses, key=lambda x: x[2])
        corrected_response = self_correct_response(query, best_response, context)
        
        # Ethical check
        is_compliant, message = check_ethical_compliance(corrected_response)
        if not is_compliant:
            return f"Sorry, I can’t provide that response due to ethical concerns ({message}). Please rephrase your question."
        
        return f"Response (from {best_source}, confidence {best_score:.2f}):\n{corrected_response}"
    
    # Fallback
    fallback_prompt = f"""
    You are Study Buddy, a peer tutor. I couldn’t find a reliable answer for "{query}". Provide a general response or suggest an alternative approach.
    """
    fallback_messages = [{"role": "system", "content": fallback_prompt}, *[{"role": msg["role"], "content": msg["content"]} for msg in context]]
    fallback_response = call_openai(fallback_messages)
    
    # Ethical check
    is_compliant, message = check_ethical_compliance(fallback_response)
    if not is_compliant:
        return f"Sorry, I can’t provide that response due to ethical concerns ({message}). Please rephrase your question."
    
    return fallback_response

# Main Chatbot
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
        context = get_optimized_context(query)
        extract_entities(query)
        response = handle_query_with_reasoning(query, chunks, embedder, index, tavily_api_key, context, user_level)
        add_to_history("assistant", response)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    run_chatbot("my_notes.md", TAVILY_API_KEY)
```

### Test Cases
- **Plan-Execute-Refine**: “Solve x^2 - 5x + 6 = 0” → Structured plan and solution.
- **Self-Correction**: “Calculate 15 * 23” → Corrects if initial answer is wrong (e.g., 325 → 345).
- **Fallback**: “Calculate invalid” → “Please clarify the calculation.”
- **Confidence-Based**: “Who is Alan Turing?” → Selects Wikipedia response with high confidence.
