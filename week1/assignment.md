# Week 1 Assignment: Study Buddy Question Answerer

[⬅️ Back to Week 1 Overview](week1.md) | [Project README](../Readme.md)

## Objective

Build a simple question-answering assistant that answers questions based only on the content of a Markdown note.

## Steps

1. **Environment Setup**
   - Create a new Python project directory.
   - Set up a virtual environment and activate it.
   - Install the following packages:
     - `langchain`
     - `python-dotenv`
     - Either `openai` or `google-generativeai` (depending on your chosen API)
   - Create a `.env` file and add your `OPENAI_API_KEY` or `GOOGLE_API_KEY`.

2. **Create a Markdown Note**
   - In your project directory, create a Markdown file (e.g., `my_notes.md`).
   - Write at least 200 words on any academic topic (e.g., "History of AI", "Principles of Thermodynamics", "Basics of Python Programming").

3. **Develop the Question Answerer**
   - Create a Python script (e.g., `study_buddy_week1.py`).
   - Load the Markdown content: Read the text from `my_notes.md` into a Python string.
   - Choose an LLM: Initialize either `ChatOpenAI` (from `langchain_openai`) or `ChatGoogleGenerativeAI` (from `langchain_google_genai`).
   - Construct a prompt template that takes the notes content and a question as input. The prompt should instruct the LLM to answer the question solely based on the provided notes content. If the answer is not found in the notes, it should state so.

   **Example Prompt:**
   ```
   You are an AI Study Assistant. Answer the following question based ONLY on the provided notes. If the answer is not in the notes, state 'I cannot find the answer in the provided notes.'

   Notes:
   {notes_content}

   Question: {question}
   Answer:
   ```

   - Allow the user to input a question.
   - Format the prompt with the notes and the user's question.
   - Send the prompt to the LLM and print the response.

4. **Test and Document**
   - Test your script with at least three different questions:
     - One question whose answer is clearly in the `my_notes.md` file.
     - One question whose answer is partially in the `my_notes.md` file.
     - One question whose answer is not in the `my_notes.md` file.
   - Include a brief `README.md` in your project folder explaining how to run your script and what each test question demonstrated.

## Deliverables

- Create a github repo which we will maintain throughout this project.
- The `my_notes.md` file you used.
- A `README.md` with setup and testing instructions.

---

[⬆️ Back to Top](#week-1-assignment-study-buddy-question-answerer) | [Week 1 Overview](week1.md) | [Project README](../Readme.md)
