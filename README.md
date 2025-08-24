# RAG Project with Gemini LLM

This project is a demonstration of a Retrieval-Augmented Generation (RAG) system built using Google's Gemini models. It provides an interactive command-line interface to manage and query multiple documents.

## What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is a technique for building powerful language model applications that can answer questions about specific information that they were not originally trained on. It combines the strengths of two different AI components:

1.  **A Retriever:** This component is responsible for finding and retrieving relevant documents from a large knowledge base (like a collection of text files, PDFs, or a database).
2.  **A Generator:** This component is a large language model (LLM), like Gemini, that takes the user's question and the retrieved documents and generates a human-like answer based on the provided information.

By combining these two, a RAG system can provide answers that are more accurate, up-to-date, and grounded in a specific set of facts, reducing the chances of the LLM "hallucinating" or making up information.

## Project Overview

This project implements an interactive, multi-document RAG pipeline. You can add documents from local files (PDF, JSON, HTML) or web URLs, and then ask questions about them. The system persists your document list between sessions.

The core components are:
-   **Data Ingestion:** Loads and parses documents from URLs (`requests`, `BeautifulSoup`), PDFs (`pypdf`), and local files (`json`, `BeautifulSoup`).
-   **Text Splitter:** Splits documents into smaller, overlapping chunks for effective retrieval.
-   **Embedding Model:** Uses Google's `models/text-embedding-004` model to convert text chunks into numerical vectors (embeddings).
-   **Vector Store:** Uses `ChromaDB` to store the embeddings in a persistent, on-disk database, allowing for efficient similarity search.
-   **LLM for Generation:** Uses Google's `gemini-1.5-flash` model to generate answers based on the user's query and the retrieved context.

## How It Works

The `rag_system.py` script provides a command-line interface with the following workflow:

1.  **Load API Key:** It loads your `GOOGLE_API_KEY` from your shell's environment variables.
2.  **Initialize:** It loads the list of previously added document sources from `document_sources.json` and initializes their corresponding vector store collections in ChromaDB.
3.  **Interactive Menu:** It presents a menu with the following options:
    -   **Add a new document:** Prompts for a file path or URL. The script then ingests the document, chunks its content, generates embeddings, and saves them to a new, persistent collection in the `chroma_db` directory.
    -   **Ask questions about a document:** Lets you select a previously loaded document. You can then ask questions in a loop. For each question, the system retrieves the most relevant passages from ChromaDB, constructs a detailed prompt, and uses the Gemini model to generate an answer based only on the provided context.
    -   **List loaded documents:** Shows all the documents that have been added.
    -   **Remove a document:** Deletes a document's vector store collection from disk and removes it from the list of sources.
    -   **Exit:** Terminates the program.

## How to Use

1.  **Set up the project:**
    If you haven't already, clone the repository and navigate into the project directory.

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    (On Windows, use `venv\Scripts\activate`)

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API key:**
    You need to set the `GOOGLE_API_KEY` environment variable. You can do this for your current session or add it to your shell's startup file (e.g., `.bashrc`, `.zshrc`).
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```
    Replace `"YOUR_API_KEY"` with your actual key.

5.  **Run the script:**
    ```bash
    python rag_system.py
    ```
    The script will start and display the main menu, ready for you to add and query documents.