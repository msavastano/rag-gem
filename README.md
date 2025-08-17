# RAG Project with Gemini LLM

This project is a demonstration of a Retrieval-Augmented Generation (RAG) system built using Google's Gemini models and the LangChain framework.

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a technique for building powerful language model applications that can answer questions about specific information that they were not originally trained on. It combines the strengths of two different AI components:

1.  **A Retriever:** This component is responsible for finding and retrieving relevant documents from a large knowledge base (like a collection of text files, PDFs, or a database).
2.  **A Generator:** This component is a large language model (LLM), like Gemini, that takes the user's question and the retrieved documents and generates a human-like answer based on the provided information.

By combining these two, a RAG system can provide answers that are more accurate, up-to-date, and grounded in a specific set of facts, reducing the chances of the LLM "hallucinating" or making up information.

## Project Overview

This project implements a simple RAG pipeline that can answer questions about Google's Gemini models. It uses a text file containing a report about Gemini as its knowledge base.

The core components are:
-   **Document Loader:** Loads the source text file.
-   **Text Splitter:** Splits the document into smaller chunks for easier processing.
-   **Embedding Model:** Uses Google's `embedding-001` model to convert the text chunks into numerical vectors (embeddings).
-   **Vector Store:** Uses FAISS, a library for efficient similarity search, to store the embeddings.
-   **LLM for Generation:** Uses Google's `gemini-pro` model to generate answers based on the retrieved information.

## How It Works

The `rag.py` script performs the following steps:

1.  **Load API Key:** It loads your `GOOGLE_API_KEY` from a `.env` file.
2.  **Load Document:** It loads the `gemini_info.txt` file using `TextLoader`.
3.  **Split Document:** It splits the loaded document into smaller chunks.
4.  **Generate Embeddings:** It uses the Gemini embedding model to create embeddings for each chunk.
5.  **Create Vector Store:** It creates an in-memory FAISS vector store and populates it with the embeddings.
6.  **Perform Similarity Search:** When you provide a query, it searches the vector store for the document chunks that are most relevant to your question.
7.  **Generate Answer:** It passes your question and the retrieved chunks to the Gemini Pro model, which then generates a final answer.

## How to Use

1.  **Set up the project:**
    If you haven't already, clone the repository and navigate into the project directory.

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    (On Windows, use `venv\\Scripts\\activate`)

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    Create a file named `.env` in the root of the project and add your Google API key to it:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY"
    ```
    Replace `"YOUR_API_KEY"` with your actual key.

5.  **Run the script:**
    ```bash
    python rag.py
    ```
    The script will run a predefined query and print the answer.

## Optional: ChromaDB (heavy, opt-in)

`chromadb` is supported by `rag_system.py` but has additional heavy dependencies (e.g. `onnxruntime`, `pulsar-client`, specific `numpy` pins) that can conflict with other packages. To avoid dependency resolution issues, install `chromadb` in a separate virtual environment only when you need it:

```bash
# create an isolated venv for chromadb
python3 -m venv .venv_chromadb
source .venv_chromadb/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install chromadb
```

Alternatively, you can install just the optional package into your main venv with:

```bash
pip install chromadb
```

If you prefer not to install `chromadb`, the project includes a FAISS-based flow (`rag.py`) which avoids ChromaDB's heavy deps and is the recommended default for local development.

## Customization

You can easily customize this project to use your own documents.
1.  Place your text file in the project directory.
2.  In `rag.py`, change the filename in the `TextLoader` from `'gemini_info.txt'` to your file's name.
