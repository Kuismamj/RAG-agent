# RAG (Retrieval-Augmented Generation) Demo

This project is a simple Retrieval-Augmented Generation (RAG) application in Python. It allows you to ask questions about your own text document using semantic search and a language model.

## How it works

1. **Text Splitting:** The program reads `oma_tiedosto.txt` and splits it into smaller chunks.
2. **Vectorization:** Each text chunk is converted into a vector (embedding) using a SentenceTransformer model.
3. **Retrieval:** When you ask a question, the program finds the most semantically similar text chunk using a FAISS index.
4. **Answer Generation:** The closest text chunk and your question are given to a Llama language model, which generates an answer.

## Usage

1. Put your text into a file named `oma_tiedosto.txt`.
2. Run the program:
   ```
   python rag.py
   ```
3. Ask questions about the document directly in the terminal.

## Dependencies

- sentence-transformers
- faiss
- numpy
- llama-cpp-python

Install dependencies:
```
pip install sentence-transformers faiss-cpu numpy llama-cpp-python
```

## Models

The program uses the Mistral 7B model (in gguf format). Download the model and place it in the `./models/` folder.
