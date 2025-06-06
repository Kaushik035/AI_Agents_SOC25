Base code for this assignment is already provided in notes.

## Assignment 2.1: Continue with StudyBuddy

update your Week 1 StudyBuddy project to implement a Retrieval-Augmented Generation (RAG) chatbot using the provided code snippet. This assignment will help you understand how to integrate a knowledge base with a language model to answer questions based on specific content.

### Evaluation Criteria:

1. **Create a Markdown Note**:
   - Write a new `my_notes.md` file (at least 1000 words) on a topic of your choice (e.g., “Introduction to Machine Learning” or “Quantum Physics Basics”).
   - Ensure the content is structured with clear sections or paragraphs.

3. **Develop the RAG Chatbot**:
   - Modify the prompt to ensure the LLM states “I cannot find the answer in the provided notes” if the retrieved chunks don’t contain relevant information (hint: add this instruction to the prompt).

4. **Test and Document**:
   - Test your chatbot with at least three questions:
     - One clearly answered in `my_notes.md`.
     - One partially answered in the notes.
     - One not covered in the notes.
   - Create a `README.md` in `week2/` explaining:
     - How to set up and run your script.
     - The three test questions, their expected outputs, and what they demonstrate.
     - Any challenges faced and how you addressed them.

Tip: Check what chunks are being retrieved by printing them out before passing them to the LLM. This will help you understand how the retrieval process works and ensure that the chunks contain relevant information.


## Assignment 2.2: Implement Another Vector Store

Currently, we are using 'faiss' as our vector store. In this assignment, you will implement another vector store of your choice and store atleast 3 pdf file using the new vector store. 

Chatbot should be able to answer questions based on the content of these pdf files and return the source of the answer (e.g., "Source: file1.pdf, page 2") in final answer.

---
