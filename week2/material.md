Below is a detailed Week 2 document for the "Study Buddy Project" curriculum, focusing on the two selected topics: **Introduction to Retrieval-Augmented Generation (RAG)** and **Introduction to Vector Databases**. The document is designed to be as comprehensive as Week 1’s `week1.md`, with a focus on manual implementation (no frameworks like LangChain), detailed explanations of inner workings, and technical resources. It builds on Week 1’s foundation of Generative AI and prompt-based chatbots, introducing students to context-aware systems using RAG and vector databases.



# Week 2: Retrieval-Augmented Generation and Vector Databases

[⬅️ Back to Project Overview](../Readme.md)

---

## Index

- [Topic 1: Introduction to Retrieval-Augmented Generation (RAG)](#topic-1-introduction-to-retrieval-augmented-generation-rag)
  - [What is RAG?](#what-is-rag)
  - [How RAG Works](#how-rag-works)
  - [Advantages of RAG](#advantages-of-rag)
  - [Technical Resource](#technical-resource-for-rag)
- [Topic 2: Introduction to Vector Databases](#topic-2-introduction-to-vector-databases)
  - [What Are Vector Databases?](#what-are-vector-databases)
  - [How Vector Databases Work](#how-vector-databases-work)
  - [Setting Up a Vector Database with FAISS](#setting-up-a-vector-database-with-faiss)
  - [Technical Resource](#technical-resource-for-vector-databases)
- [Step 3: Building a RAG-Based Study Buddy](#step-3-building-a-rag-based-study-buddy)
  - [Step 3.1: Prepare and Chunk Documents](#step-31-prepare-and-chunk-documents)
  - [Step 3.2: Generate Vector Embeddings](#step-32-generate-vector-embeddings)
  - [Step 3.3: Store Embeddings in FAISS](#step-33-store-embeddings-in-faiss)
  - [Step 3.4: Retrieve Relevant Documents](#step-34-retrieve-relevant-documents)
  - [Step 3.5: Integrate RAG with the LLM](#step-35-integrate-rag-with-the-llm)
  - [Step 3.6: Build the Interactive Chatbot](#step-36-build-the-interactive-chatbot)
  - [Tips for Better RAG Implementation](#tips-for-better-rag-implementation)
- [Assignment](#assignment)
- [Bonus: Understanding Embeddings Visually](#bonus-understanding-embeddings-visually)

---

## Topic 1: Introduction to Retrieval-Augmented Generation (RAG)

### What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of large language models (LLMs) by combining them with an external retrieval system. Unlike standalone LLMs that rely solely on their internal knowledge (learned during training), RAG enables the model to fetch relevant information from a collection of documents before generating a response. This makes RAG ideal for tasks where answers need to be grounded in specific, user-provided data, such as answering questions based on lecture notes or textbooks.

#### Example:
- **Without RAG**: An LLM might answer “What is the capital of France?” with “Paris” based on its training data, but it could struggle with questions about your personal notes (e.g., “What did my AI lecture notes say about transformers?”).
- **With RAG**: The system retrieves relevant sections from your notes and uses them to craft a precise, context-aware response.

For the Study Buddy project, RAG allows your chatbot to answer questions directly from your Markdown notes, ensuring accuracy and relevance.

### How RAG Works

RAG combines two main components: a **retriever** and a **generator**. Here’s a step-by-step breakdown of the process:

1. **Document Preparation**:
   - Your documents (e.g., Markdown notes) are split into smaller chunks (e.g., paragraphs or sentences) to make retrieval manageable.
   - Each chunk is converted into a numerical representation called a **vector embedding**, which captures its semantic meaning.

2. **Retrieval**:
   - When a user asks a question, the question is also converted into a vector embedding.
   - A similarity search (e.g., cosine similarity) compares the question’s embedding to the document chunks’ embeddings to find the most relevant chunks.

3. **Generation**:
   - The retrieved chunks are included in the prompt sent to the LLM, providing context.
   - The LLM generates a response based on both the question and the retrieved content, ensuring the answer is grounded in the provided documents.

#### Inner Workings:
- **Vector Embeddings**: These are high-dimensional vectors (e.g., 768-dimensional arrays) that represent text in a numerical format. Words or sentences with similar meanings have similar vectors, allowing the system to find relevant content via mathematical comparisons.
- **Similarity Search**: The retriever calculates the similarity between the question’s embedding and document embeddings, often using cosine similarity (a measure of how closely aligned two vectors are).
- **Prompt Construction**: The retrieved chunks are injected into the prompt, e.g., “Based on the following notes: [chunk], answer: [question].” This guides the LLM to focus on the provided context.

### Advantages of RAG

1. **Contextual Accuracy**: RAG ensures responses are based on specific, user-provided data, reducing the risk of hallucination (when LLMs generate incorrect or invented information).
2. **Up-to-Date Information**: By retrieving from your own documents, RAG can incorporate recent or personal data that the LLM wasn’t trained on.
3. **Scalability**: RAG works with large document sets, as the retriever efficiently narrows down relevant content.
4. **Flexibility**: It can be applied to various tasks, like question answering, summarization, or note-based tutoring.

## Topic 2: Introduction to Vector Databases

### What Are Vector Databases?

A vector database is a specialized database designed to store, manage, and query high-dimensional vector embeddings efficiently. Unlike traditional databases (e.g., SQL for structured data like tables), vector databases are optimized for numerical vectors, enabling fast similarity searches to find text with similar meanings.

#### Example:
- If your notes discuss “machine learning” and “deep learning,” a vector database can store their embeddings and quickly find related content when you query “What is AI?”

For the Study Buddy project, a vector database will store embeddings of your Markdown notes, allowing the RAG system to retrieve relevant sections for answering questions.

### How Vector Databases Work

Vector databases rely on two key concepts: **embeddings** and **similarity search**. Here’s how they function:

1. **Embedding Generation**:
   - Text (e.g., a sentence or paragraph) is converted into a vector using a model like `sentence-transformers`. For example, “AI is powerful” might become a 768-dimensional vector like `[0.12, -0.45, 0.89, ...]`.
   - These vectors capture semantic meaning, so similar texts (e.g., “AI is transformative”) have similar vectors.

2. **Storage**:
   - The vectors are stored in the database, often with metadata (e.g., the original text or document ID).
   - The database uses indexing techniques (e.g., Approximate Nearest Neighbors) to organize vectors for fast retrieval.

3. **Querying**:
   - A query (e.g., “What is AI?”) is converted into a vector.
   - The database performs a similarity search to find the closest vectors (i.e., the most relevant document chunks) using metrics like cosine similarity or Euclidean distance.

#### Inner Workings:
- **Indexing**: Vector databases use algorithms like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File) to index vectors, enabling fast searches even with millions of entries.
- **Cosine Similarity**: This metric measures the angle between two vectors. If the angle is small (cosine close to 1), the texts are semantically similar. For example:
  - Vector A (for “AI is powerful”): `[0.12, -0.45, 0.89, ...]`
  - Vector B (for “What is AI?”): `[0.10, -0.40, 0.87, ...]`
  - Cosine similarity: `cos(θ) = (A·B)/(||A||·||B||)`, where a value near 1 indicates high similarity.
- **Scalability**: Techniques like quantization (reducing vector precision) or clustering make searches faster and less memory-intensive.

### Setting Up a Vector Database with FAISS

For this course, we’ll use **FAISS** (Facebook AI Similarity Search), a lightweight, open-source library for vector storage and retrieval. FAISS is ideal for small-scale projects and runs locally without external dependencies.

#### Why FAISS?
- It’s free, efficient, and beginner-friendly.
- It supports exact and approximate nearest neighbor searches.
- It integrates easily with Python and `sentence-transformers`.

#### Example Workflow:
1. Convert document chunks to embeddings using `sentence-transformers`.
2. Store embeddings in a FAISS index.
3. Query the index with a question’s embedding to retrieve the top-k relevant chunks.

### Must Read Resources and Documentation (specially if above explaination jumped over your head or you want to learn more)
- [Building RAG Systems](https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-1-with-implementations/) - A beginner-friendly introduction to RAG systems, covering the basics
- [Advanced RAG with FAISS](https://machinelearningmastery.com/advanced-techniques-to-build-your-rag-system/?ref=dailydev) – A comprehensive guide on building RAG systems using FAISS, including practical examples.
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki) – Official guide for setting up and using FAISS. Focus on the “Getting Started” section.
- [Sentence Transformers Documentation](https://www.sbert.net/) – Explains how to generate embeddings with `sentence-transformers`.

---

## Step 3: Building a RAG-Based Study Buddy

In Week 1, you built a simple chatbot using the `transformers` library and GPT-2, answering questions with a basic prompt. In Week 2, you’ll upgrade it to a RAG-based system that retrieves relevant sections from your Markdown notes before generating answers. We’ll avoid frameworks like LangChain and manually implement the RAG pipeline using `sentence-transformers` for embeddings, FAISS for the vector database, and GPT-2 for generation.

Here’s the detailed process, broken into steps, with code and explanations.

---

### The RAG Pipeline

To build a RAG-based Study Buddy, follow these steps:
1. **Prepare and Chunk Documents**: Load and split your Markdown notes into manageable pieces.
2. **Generate Vector Embeddings**: Convert document chunks into vectors using `sentence-transformers`.
3. **Store Embeddings in FAISS**: Create a FAISS index to store and query embeddings.
4. **Retrieve Relevant Documents**: Search the FAISS index to find chunks relevant to the user’s question.
5. **Integrate RAG with the LLM**: Combine retrieved chunks with the question in a prompt for GPT-2.
6. **Build the Interactive Chatbot**: Create a command-line interface for users to ask questions and get answers.

Let’s dive into each step.

---

### Step 3.1: Prepare and Chunk Documents

To make retrieval efficient, we split large documents into smaller chunks (e.g., paragraphs or sentences). This ensures the system retrieves only the most relevant pieces.

#### Instructions:
- Load your Markdown notes (e.g., `my_notes.md`) into a Python string.
- Split the text into chunks (e.g., by paragraphs or fixed-length sentences).
- Store chunks in a list for embedding.

#### Code Example:
```python
def load_and_chunk_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split by paragraphs (double newlines)
    chunks = text.split('\n\n')
    # Clean and filter chunks
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]
    return chunks
```

#### Explanation:
- **Loading**: Reads the Markdown file into a string.
- **Chunking**: Splits by double newlines (`\n\n`), which typically separate paragraphs in Markdown. Adjust the splitting logic (e.g., by sentences) based on your needs.
- **Cleaning**: Removes empty or very short chunks to avoid noise.

---

### Step 3.2: Generate Vector Embeddings

Each chunk needs to be converted into a vector embedding using a model like `sentence-transformers`.

#### Instructions:
- Install `sentence-transformers`: `pip install sentence-transformers`.
- Load a pre-trained model (e.g., `all-MiniLM-L6-v2`, which is lightweight and effective).
- Generate embeddings for each chunk.

#### Code Example:
```python
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks):
    # Load the embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings (returns a list of numpy arrays)
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings
```

#### Explanation:
- **Model**: `all-MiniLM-L6-v2` produces 384-dimensional embeddings, balancing performance and size.
- **Encoding**: The `encode` method converts each chunk into a vector. The output is a numpy array of shape `(num_chunks, embedding_dim)`.
- **Why Sentence Transformers?**: Unlike word-level embeddings (e.g., Word2Vec), sentence transformers generate embeddings for entire sentences or paragraphs, capturing semantic meaning.

---

### Step 3.3: Store Embeddings in FAISS

Use FAISS to store embeddings and enable fast similarity searches.

#### Instructions:
- Install FAISS: `pip install faiss-cpu` (use `faiss-gpu` if you have a compatible GPU).
- Create a FAISS index and add embeddings.
- Store the original chunks alongside embeddings for retrieval.

#### Code Example:
```python
import faiss
import numpy as np

def create_faiss_index(embeddings):
    # Get embedding dimension
    dimension = embeddings.shape[1]
    # Create a flat index with L2 distance
    index = faiss.IndexFlatL2(dimension)
    # Add embeddings to the index
    index.add(embeddings)
    return index
```

#### Explanation:
- **IndexFlatL2**: A simple FAISS index that uses Euclidean distance (L2) for similarity. For small datasets, this is sufficient; for larger ones, consider `IndexIVFFlat` for faster searches.
- **Adding Embeddings**: The `add` method stores the embeddings in the index, enabling efficient querying.
- **Storage**: We’ll keep the original chunks in a list to map back to the text after retrieval.

---

### Step 3.4: Retrieve Relevant Documents

When the user asks a question, convert it to an embedding and search the FAISS index for the most relevant chunks.

#### Instructions:
- Convert the question to an embedding using the same `sentence-transformers` model.
- Query the FAISS index to get the top-k relevant chunks.
- Return the corresponding text chunks.

#### Code Example:
```python
def retrieve_chunks(question, embedder, index, chunks, k=2):
    # Convert question to embedding
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    # Search for top-k similar chunks
    distances, indices = index.search(question_embedding, k)
    # Get the corresponding text chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks
```

#### Explanation:
- **Query Embedding**: The question is encoded into a vector using the same model as the document chunks.
- **Search**: The `search` method returns the indices of the top-k closest embeddings and their distances.
- **Retrieval**: We map the indices back to the original text chunks for use in the prompt.

---

### Step 3.5: Integrate RAG with the LLM

Combine the retrieved chunks with the user’s question in a prompt and pass it to GPT-2 for generation.

#### Instructions:
- Create a prompt that includes the retrieved chunks as context and the user’s question.
- Use the Week 1 GPT-2 pipeline to generate a response.
- Process the output to extract the assistant’s answer.

#### Code Example:
```python
from transformers import pipeline

def generate_rag_response(question, relevant_chunks, generator):
    # Create the prompt
    context = "\n".join(relevant_chunks)
    prompt = f"Based on the following notes:\n{context}\n\nQuestion: {question}\nAnswer:"
    # Generate response
    response = generator(prompt, max_length=100, num_return_sequences=1)
    generated_text = response[0]["generated_text"]
    # Extract the answer
    answer = generated_text.split("Answer:")[1].strip()
    if not answer.endswith("."):
        last_period = answer.rfind(".")
        if last_period != -1:
            answer = answer[:last_period + 1]
    return answer
```

#### Explanation:
- **Prompt Structure**: The prompt includes the context (retrieved chunks) and the question, instructing the LLM to base its answer on the provided notes.
- **Generation**: Uses the same GPT-2 pipeline from Week 1, with a longer `max_length` to accommodate the context.
- **Processing**: Extracts the answer part and ensures it ends cleanly.

---

### Step 3.6: Build the Interactive Chatbot

Tie everything together into a command-line chatbot that takes user questions, retrieves relevant chunks, and generates answers.

#### Instructions:
- Load the document, embedder, FAISS index, and GPT-2 model once at startup.
- Create a loop to accept user questions, retrieve chunks, and generate responses.
- Add an exit condition (e.g., typing “quit”).

#### Full Code Example:
```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load and chunk document
def load_and_chunk_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    chunks = text.split('\n\n')
    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]
    return chunks

# Step 2: Generate embeddings
def generate_embeddings(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return embeddings, embedder

# Step 3: Create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 4: Retrieve relevant chunks
def retrieve_chunks(question, embedder, index, chunks, k=2):
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    distances, indices = index.search(question_embedding, k)
    return [chunks[idx] for idx in indices[0]]

# Step 5: Generate RAG response
def generate_rag_response(question, relevant_chunks, generator):
    context = "\n".join(relevant_chunks)
    prompt = f"Based on the following notes:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    generated_text = response[0]["generated_text"]
    answer = generated_text.split("Answer:")[1].strip()
    if not answer.endswith("."):
        last_period = answer.rfind(".")
        if last_period != -1:
            answer = answer[:last_period + 1]
    return answer

# Main chatbot loop
def run_rag_chatbot(file_path):
    # Initialize components
    print("Loading document and models...")
    chunks = load_and_chunk_document(file_path)
    embeddings, embedder = generate_embeddings(chunks)
    index = create_faiss_index(embeddings)
    generator = pipeline("text-generation", model="gpt2")
    
    print("Welcome to the RAG Study Buddy! Type 'quit' to exit.")
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            print("Goodbye!")
            break
        relevant_chunks = retrieve_chunks(question, embedder, index, chunks)
        answer = generate_rag_response(question, relevant_chunks, generator)
        print(f"Assistant: {answer}")

# Start the chatbot
run_rag_chatbot("my_notes.md")
```

#### Explanation:
- **Modular Design**: Each function handles a specific part of the RAG pipeline, making it easy to debug or modify.
- **Initialization**: Loads the document, embedder, index, and generator once to save time.
- **Interaction**: Users input questions, and the system retrieves relevant chunks and generates answers until they quit.
- **Running**: Save as `rag_chatbot.py` and run with `python rag_chatbot.py`, assuming `my_notes.md` exists and dependencies are installed.

#### Setup Instructions:
1. Install dependencies: `pip install transformers sentence-transformers faiss-cpu torch`.
2. Create a `my_notes.md` file with at least 200 words on an academic topic.
3. Run the script and test with questions related to your notes.

---

### Tips for Better RAG Implementation

- **Chunk Size**: Experiment with chunk sizes (e.g., 100 words vs. paragraphs) to balance retrieval accuracy and context length.
- **Top-k Retrieval**: Adjust the number of retrieved chunks (`k`) based on your needs; more chunks provide more context but may overwhelm the LLM.
- **Embedding Model**: Try other `sentence-transformers` models (e.g., `all-distilroberta-v1`) for better embeddings at the cost of speed.
- **Error Handling**: Add checks for empty questions or missing documents.
- **Performance**: For large document sets, consider FAISS’s `IndexIVFFlat` for faster searches (requires additional setup).

---

## Bonus: Understanding Embeddings Visually

To better grasp vector embeddings, check out this resource:
- [Visualizing Word Embeddings](https://www.tensorflow.org/tutorials/text/word_embeddings) – A TensorFlow tutorial that explains embeddings with interactive visualizations. Focus on the “What are embeddings?” section for intuition.

For a hands-on exercise:
- Use `sentence-transformers` to generate embeddings for two sentences (e.g., “AI is powerful” and “Machine learning is transformative”).
- Compute their cosine similarity manually using NumPy:
  ```python
  from sentence_transformers import SentenceTransformer
  import numpy as np

  embedder = SentenceTransformer('all-MiniLM-L6-v2')
  sentences = ["AI is powerful", "Machine learning is transformative"]
  embeddings = embedder.encode(sentences)
  cos_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
  print(f"Cosine Similarity: {cos_sim}")
  ```
- A value close to 1 indicates high semantic similarity.

---

[⬆️ Back to Top](#week-2-retrieval-augmented-generation-and-vector-databases) | [Project README](../Readme.md) | [Assignment](assignment.md)

