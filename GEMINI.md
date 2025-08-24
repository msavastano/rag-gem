# Gemini RAG Project Context

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) system that uses the Gemini Large Language Model to answer questions about a collection of documents. It's a command-line application written in Python that allows users to add, manage, and query documents from local files (PDF, JSON, HTML) and web URLs.

The core components of the project are:

*   **Data Ingestion:** Loads and parses documents from various sources.
*   **Text Splitting:** Splits documents into smaller, overlapping chunks for effective retrieval.
*   **Embedding Model:** Uses Google's `models/text-embedding-004` to convert text chunks into vector embeddings.
*   **Vector Store:** Uses `ChromaDB` to store and retrieve the embeddings.
*   **Generative Model:** Uses Google's `gemini-1.5-flash` to generate answers based on user queries and retrieved context.

The project is structured as a single Python script, `rag_system.py`, which contains all the logic for the RAG pipeline. The script uses a `document_sources.json` file to persist the list of documents between sessions.

## Building and Running

To build and run the project, follow these steps:

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your API key:**
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

4.  **Run the script:**
    ```bash
    python rag_system.py
    ```

## Development Conventions

*   **Code Style:** The code follows the PEP 8 style guide for Python.
*   **Dependencies:** Project dependencies are managed using a `requirements.txt` file.
*   **Modularity:** The code is organized into logical modules within the `rag_system.py` script:
    *   Data Ingestion
    *   Embedding and Indexing
    *   Retrieval and Generation
    *   Main Execution Block
*   **Error Handling:** The code includes error handling for file I/O, network requests, and other potential issues.
*   **Persistence:** The list of document sources is persisted in a JSON file named `document_sources.json`.
*   **Vector Store:** The ChromaDB vector store is persisted in the `chroma_db` directory.
