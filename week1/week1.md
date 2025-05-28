# Week 1: Generative AI and Its Foundations

[⬅️ Back to Project Overview](../Readme.md)

Welcome to Week 1 of the **Study Buddy Project**! This week, we’re embarking on an exciting journey into the world of Generative AI. Whether you’re new to AI or looking to deepen your understanding, this week will lay the groundwork for building your own AI-powered study assistant. We’ll explore what Generative AI is, how it works, its advantages, and how to use it in practical applications. You’ll also get hands-on with tools like LangChain, local language models, and the OpenAI API.

By the end of this week, you’ll have a solid grasp of Generative AI basics and be ready to start experimenting with your own projects. Let’s get started!

---

## Topic 1: What is Generative AI?

**Generative AI** is a type of artificial intelligence that creates new content—like text, images, or music—based on patterns it learns from existing data. Unlike traditional AI, which might classify data (e.g., spotting spam emails), Generative AI generates entirely new outputs that mimic the style or structure of its training data.

#### Examples:
- **Text**: Writing stories, answering questions, or generating code.
- **Images**: Creating artwork or photos from text descriptions.
- **Music**: Composing songs in different genres.

This ability to “generate” makes it a powerful tool for creativity and automation, transforming how we solve problems and build applications.

**Resource**:  
- [Generative AI: A Creative New World](https://www.sequoiacap.com/article/generative-ai-a-creative-new-world/) – A beginner-friendly overview of Generative AI and its impact.

---

## Topic 2: How Does Generative AI Work?

Generative AI, especially for text, relies on a technology called **transformers**, a type of neural network designed to handle sequential data like sentences or paragraphs. Let’s break it down step by step:

#### Key Concepts:
1. **Transformers**:  
   - Transformers use a mechanism called **attention** to figure out which words in a sentence matter most to each other. For example, in “The cat sat on the mat,” the model understands that “cat” and “sat” are closely related.
   - This helps the AI generate text that makes sense in context.

2. **Training**:  
   - The model is trained on massive datasets—like books, websites, or articles. It learns by predicting the next word in a sequence. For example, given “The sky is,” it might predict “blue.”
   - Over time, it builds an understanding of language patterns.

3. **Generation**:  
   - Once trained, the model generates text one word at a time. You give it a starting point (a “prompt”), and it predicts what comes next, building a response piece by piece.
   - Example: Input “What is the capital” → Output “of France? Paris.”

This process allows Generative AI to produce coherent, human-like outputs, making it ideal for tasks like writing or answering questions.

**Resource**: 
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) – A visual guide to how transformers work, perfect for beginners.

**Video**:  (SKIP IF YOU KNOW HOW TRANSFORMER WORKS OR READ ABOVE ARTICLE COMPLETLY)
- [Transformers Explained](https://www.youtube.com/watch?v=TQQlZhbC5ps) – A clear, beginner-friendly video by StatQuest (17:45).





## Topic 3: Advantages of Generative AI Over Other AI Approaches

Generative AI stands out from traditional AI methods due to its ability to create, adapt, and scale in ways that other approaches cannot. Here’s why it’s a game-changer:

### Key Advantages:

1. **Creativity and Novelty**  
   - Unlike traditional AI, which focuses on classification or prediction, Generative AI can produce entirely new content—think stories, artwork, or innovative solutions. This makes it a go-to for creative industries and problem-solving.

2. **Automation of Complex Tasks**  
   - It can handle tasks requiring human-like understanding, such as drafting text or generating code, streamlining workflows and reducing manual effort.

3. **Adaptability and Flexibility**  
   - A single generative model can tackle diverse tasks—answering questions, summarizing documents, or translating languages—based on the input it receives.

4. **Scalability**  
   - Once trained, these models can process countless requests efficiently, perfect for applications like chatbots or large-scale content creation.

5. **Personalization**  
   - Generative AI can customize outputs to match user preferences, improving experiences in areas like education or recommendations.

### Example:
- **Traditional AI**: A spam filter labels emails as spam or not using predefined rules.  
- **Generative AI**: A chatbot crafts natural, context-aware replies to customer inquiries.
---

## Step 4: Integrate with Your Application

Once you’ve chosen a use case (a chatbot), selected a model (GPT-2), and set up your environment (Python with `transformers` and `torch`), it’s time to integrate the Generative AI model into your application. This step is where the magic happens—you’ll connect the AI to your code and make it functional for users. We’ll focus on building a simple chatbot that takes user input and generates responses, all without relying on frameworks like LangChain. Instead, we’ll use the Hugging Face `transformers` library directly to handle the model and write custom logic to process inputs and outputs.

Here’s the detailed process, broken into manageable parts, with explanations and code to guide you.

---

### The Integration Process

To integrate Generative AI into your application, follow these steps:
1. **Load the Generative Model**: Import and initialize the model using the `transformers` library.
2. **Design the Input Structure**: Prepare user input in a format the model understands (e.g., a prompt).
3. **Generate the Output**: Use the model to create a response based on the input.
4. **Process the Output**: Clean up the generated text for display or further use.
5. **Build a Simple Interface**: Create a basic loop or function to interact with users.

Let’s explore each step with our chatbot example.

---

### Step 4.1: Load the Generative Model

The first task is to load the GPT-2 model into your application. The `transformers` library makes this straightforward by providing a `pipeline` utility, which abstracts much of the complexity of tokenization and inference.

#### Instructions:
- Use the `pipeline` class from `transformers` to load GPT-2 for text generation.
- Specify `"text-generation"` as the task and `"gpt2"` as the model name.
- Store the loaded model in a variable for reuse.

#### Code Example:
```python
from transformers import pipeline

# Load GPT-2 for text generation
generator = pipeline("text-generation", model="gpt2")
```

#### Explanation:
- **Pipeline**: The `pipeline` is a high-level interface that handles model loading, tokenization, and inference in one go. For beginners, it’s an excellent way to get started without diving into low-level details.
- **GPT-2**: We’re using the base version of GPT-2, which is small enough to run on most machines but still capable of generating coherent text.
- **Why Not LangChain?**: LangChain would handle model loading and context management automatically, but here we’re doing it manually to understand the process fully.

Once loaded, `generator` is ready to take input and produce text. This step only needs to happen once in your application.

---

### Step 4.2: Design the Input Structure

Generative models like GPT-2 don’t inherently “know” what you want—they generate text based on the input you provide, called a **prompt**. A well-crafted prompt guides the model to produce relevant responses.

#### Instructions:
- Create a prompt that includes the user’s input and hints at the desired output (e.g., an assistant’s response).
- Use clear separators (like “User:” and “Assistant:”) to structure the prompt.
- Keep the prompt concise but informative.

#### Code Example:
```python
def create_prompt(user_input):
    prompt = f"User: {user_input}\nAssistant:"
    return prompt
```

#### Explanation:
- **Prompt Structure**: The format `User: [input]\nAssistant:` tells the model that what follows “Assistant:” should be a response to the user’s question. The newline (`\n`) helps the model understand the transition.
- **Customization**: For our chatbot, this simple structure works. If you wanted a more specific tone (e.g., formal or funny), you could add instructions like “You are a friendly AI expert” to the prompt.
- **Manual Approach**: Without LangChain, we’re responsible for designing the prompt ourselves, rather than relying on a framework to manage context or memory.

For example, if the user asks, “What is AI?”, the prompt becomes:
```
User: What is AI?
Assistant:
```

---

### Step 4.3: Generate the Output

With the prompt ready, it’s time to generate a response using the model. This is where the AI does its work—taking the prompt and extending it with new text.

#### Instructions:
- Pass the prompt to the `generator` object.
- Set parameters like `max_length` to control the response length and `num_return_sequences` to specify how many responses to generate (we’ll use 1).
- Extract the generated text from the output.

#### Code Example:
```python
def generate_response(prompt):
    # Generate text with GPT-2
    response = generator(prompt, max_length=50, num_return_sequences=1)
    # Extract the generated text
    generated_text = response[0]["generated_text"]
    return generated_text
```

#### Explanation:
- **Generation**: The `generator` takes the prompt and continues it, producing text up to 50 tokens (words or punctuation marks). Adjust `max_length` based on how detailed you want the response.
- **Output Format**: The `generator` returns a list of dictionaries, where each dictionary contains a `generated_text` key. Since we set `num_return_sequences=1`, we only get one response.
- **Parameters**: You can tweak settings like `temperature` (for creativity) or `top_k` (for word choice), but we’ll keep it simple for now.

For our prompt `User: What is AI?\nAssistant:`, the model might output:
```
User: What is AI?
Assistant: AI, or Artificial Intelligence, is the simulation of human intelligence in machines that are programmed to think and learn.
```

---

### Step 4.4: Process the Output

The raw output from GPT-2 includes the prompt itself, which we don’t want to show users. We need to clean it up to display only the assistant’s response.

#### Instructions:
- Split the generated text at “Assistant:” and take the second part.
- Strip any extra whitespace or incomplete sentences.
- Handle edge cases (e.g., if the model doesn’t finish the response).

#### Code Example:
```python
def process_response(generated_text):
    # Split at "Assistant:" and take the response part
    assistant_response = generated_text.split("Assistant:")[1].strip()
    # Optional: Cut off incomplete sentences (simple approach)
    if not assistant_response.endswith("."):
        last_period = assistant_response.rfind(".")
        if last_period != -1:
            assistant_response = assistant_response[:last_period + 1]
    return assistant_response
```

#### Explanation:
- **Splitting**: We discard the prompt and keep only what follows “Assistant:”.
- **Cleaning**: The `strip()` removes leading/trailing whitespace. The optional cutoff ensures the response ends cleanly, though GPT-2 often completes thoughts well.
- **Manual Effort**: Without LangChain, we handle post-processing ourselves, giving us full control over the output.

For the example output above, the processed response becomes:
```
AI, or Artificial Intelligence, is the simulation of human intelligence in machines that are programmed to think and learn.
```

---

### Step 4.5: Build a Simple Interface

Finally, tie everything together into a basic application. For our chatbot, we’ll create a command-line interface where users can type questions and get responses.

#### Instructions:
- Combine the previous functions into a main loop.
- Prompt the user for input, generate a response, and display it.
- Add a way to exit the loop (e.g., typing “quit”).

Here’s the full code integrating all the steps:

```python
from transformers import pipeline

# Load the model
generator = pipeline("text-generation", model="gpt2")

# Create the prompt
def create_prompt(user_input):
    return f"User: {user_input}\nAssistant:"

# Generate the response
def generate_response(prompt):
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]

# Process the response
def process_response(generated_text):
    assistant_response = generated_text.split("Assistant:")[1].strip()
    if not assistant_response.endswith("."):
        last_period = assistant_response.rfind(".")
        if last_period != -1:
            assistant_response = assistant_response[:last_period + 1]
    return assistant_response

# Main chatbot loop
def run_chatbot():
    print("Welcome to the AI Chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        prompt = create_prompt(user_input)
        raw_response = generate_response(prompt)
        final_response = process_response(raw_response)
        print(f"Assistant: {final_response}")

# Start the chatbot
run_chatbot()
```

#### Explanation:
- **Modular Functions**: Each function handles one part of the process, making the code reusable and easy to debug.
- **Loop**: The `run_chatbot` function creates an interactive experience, asking for input and displaying responses until the user quits.
- **Output**: For “What is AI?”, the chatbot might say: “Assistant: AI, or Artificial Intelligence, is the simulation of human intelligence in machines.”

#### Running the Code:
1. Save it as `chatbot.py`.
2. Run it with `python chatbot.py` in your terminal (assuming your environment is set up).
3. Type questions and watch the chatbot respond!

---

### Tips for Better Integration (will learn all these things further in course)

- **Prompt Engineering**: Experiment with prompts like “You are a helpful AI. Answer: [user_input]” for better results.
- **Model Parameters**: Try `temperature=0.7` for more creative responses or `max_length=100` for longer answers.
- **Error Handling**: Add checks for empty inputs or model failures.
- **Scalability**: For a web app, replace the command-line loop with a Flask or FastAPI endpoint.

---


This chatbot is just the beginning. With these skills, you can extend it to handle multi-turn conversations, integrate it into a website, or adapt it for other use cases like content generation. Happy coding!

```python
from transformers import pipeline

# Load the model
generator = pipeline("text-generation", model="gpt2")

# Create the prompt
def create_prompt(user_input):
    return f"User: {user_input}\nAssistant:"

# Generate the response
def generate_response(prompt):
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]["generated_text"]

# Process the response
def process_response(generated_text):
    assistant_response = generated_text.split("Assistant:")[1].strip()
    if not assistant_response.endswith("."):
        last_period = assistant_response.rfind(".")
        if last_period != -1:
            assistant_response = assistant_response[:last_period + 1]
    return assistant_response

# Main chatbot loop
def run_chatbot():
    print("Welcome to the AI Chatbot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        prompt = create_prompt(user_input)
        raw_response = generate_response(prompt)
        final_response = process_response(raw_response)
        print(f"Assistant: {final_response}")

# Start the chatbot
run_chatbot()

```
---


## 4. Introduction to Better LLMs

Not all LLMs are the same—some are more advanced, offering improved accuracy, richer responses, and better context understanding. These "better" LLMs typically have:
- **Larger Model Sizes:** More parameters (e.g., billions) mean they can handle complex language patterns.
- **More Training Data:** Exposure to diverse datasets makes them versatile across topics.
- **Advanced Architectures:** New designs improve how they process and generate text.

You can access these models in two ways: running them **locally** on your machine or using **API-based services**. Each approach has its pros and cons:
- **Local Models:** Full control, no recurring costs, but you’ll need powerful hardware.
- **API-based Models:** Easy to use, access to cutting-edge tech, but they come with usage fees and reliance on external services.

Let’s explore both options.

---

### 3.1. Using Local LLMs

Running LLMs locally gives you privacy and customization options, ideal if you want to keep everything in-house. Here’s how to get started.
(ANDDDD THERE ARE FREEEEE)
### Popular Local LLMs
- **Llama 2:** A robust open-source model from Meta, available in sizes like 7B, 13B, or 70B parameters.
- **Mistral:** Efficient and high-performing, great for local setups.
- **GPT-Neo/GPT-J:** Open-source alternatives to GPT-3, less demanding on resources.

### Setting It Up
You’ll need:
- **Hardware:** A strong GPU (e.g., NVIDIA RTX 3090 with 16GB+ VRAM) and plenty of RAM.
- **Software:** Python with libraries like `transformers` and `torch`.

Here’s an example using Llama 2:
1. **Install Libraries:**
   ```bash
   pip install transformers torch
   ```
2. **Load the Model:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_name = "meta-llama/Llama-2-7b-hf"  # Path to your model
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name)

   def generate_response(prompt):
       inputs = tokenizer(prompt, return_tensors="pt")
       outputs = model.generate(inputs.input_ids, max_length=100)
       return tokenizer.decode(outputs[0], skip_special_tokens=True)
   ```


   ORRRRRR
   I RECOMMEND using ollama, install thier CLI, pull models by one command
   https://ollama.com/library/mistral

### Challenges
- **Resource Needs:** Larger models demand more VRAM (e.g., 14GB for Llama 2 7B).
- **Speed:** Slower on weaker hardware compared to APIs.
- **Tip:** Use quantized models (e.g., 8-bit) to save memory, though performance might dip slightly.

---

## 3. Using API-based LLMs

If you’d rather skip the hardware hassle, API-based LLMs offer top-tier performance with minimal setup. Let’s look at two popular services.

### Popular API Services
- **OpenAI’s GPT-4:** Exceptional at understanding and generating text, widely used.
- **Anthropic’s Claude:** Strong performance with a focus on safety and ethics.

### Setting It Up
You’ll need an API key and a simple library. Here’s an example with OpenAI:
1. **Install the Library:**
   ```bash
   pip install openai
   ```
2. **Generate Responses:**
   ```python
   import openai

   openai.api_key = "your-api-key-here"

   def generate_response(prompt):
       response = openai.ChatCompletion.create(
           model="gpt-4",
           messages=[{"role": "user", "content": prompt}]
       )
       return response.choices[0].message.content
   ```

### Pros and Cons
- **Pros:** No hardware needed, instant access to the latest models.
- **Cons:** Costs scale with usage, and your data goes to the provider’s servers. [NO worries, i will give you guys openai api for this project with some weekly use limit, just dm me.]


## Assignment

Checkout [assignment.md](assignment.md) for this week’s hands-on tasks and exercises. The assignment will help you reinforce your understanding of Generative AI concepts and guide you through building and experimenting with your own AI-powered chatbot. Make sure to complete the exercises and submit your solutions as instructed in the assignment file.

---

## Bonus: Basic Prompt Engineering (USEFUL)

**[Google's 9 Hour AI Prompt Engineering Course In 20 Minutes (YouTube, Tina Huang)](https://www.youtube.com/watch?v=p09yRj47kNM&ab_channel=TinaHuang)**

Why spend 9 hours when you can get the gist in 20 minutes? Now that's what I call prompt engineering! 😄

---

[⬆️ Back to Top](#week-1-generative-ai-and-its-foundations) | [Project README](../Readme.md) | [Assignment](assignment.md)
