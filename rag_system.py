import os
import re
import hashlib
import google.generativeai as genai
import json
from bs4 import BeautifulSoup
import requests
from pypdf import PdfReader

# Try to import chromadb optionally. Installing chromadb can pull heavy
# dependencies (onnxruntime, pulsar-client, numpy pins) which may conflict
# in some environments. Make it optional and show a clear message when not
# available so users can choose to install it in a clean venv or use the
# FAISS-based pipeline instead.
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except Exception:
    chromadb = None
    embedding_functions = None
    CHROMADB_AVAILABLE = False

# --- Configuration ---
# Ensure the GOOGLE_API_KEY environment variable is set.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit()
genai.configure(api_key=GOOGLE_API_KEY)

# Constants for models and database
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_GENERATIVE_MODEL = "gemini-2.5-flash"
DB_PATH = "./chroma_db"
COLLECTION_NAME_PREFIX = "rag_collection_"

# --- Module 1: Data Ingestion ---

def scrape_url(url):
    """
    Scrapes the text content from a single URL.
    Args:
        url (str): The URL to scrape.
    Returns:
        str: The extracted text from the URL, or None if scraping fails.
    """
    try:
        print(f"Scraping URL: {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return None

def load_and_chunk_data(source, chunk_size=1000, chunk_overlap=100):
    """
    Loads data from a file or URL, extracts text, and splits it into chunks.
    Args:
        source (str): The path to the file or the URL to scrape.
        chunk_size (int): The maximum size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.
    Returns:
        list[str]: A list of text chunks, or None if processing fails.
    """
    text = ""
    is_url = source.startswith('http://') or source.startswith('https://')

    if is_url:
        text = scrape_url(source)
        if not text:
            return None
    else:  # It's a file path
        if not os.path.exists(source):
            print(f"Error: File not found at {source}")
            return None

        try:
            print(f"Loading and processing file: {source}...")
            file_extension = os.path.splitext(source)[1].lower()

            if file_extension == '.json':
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, indent=2)
            elif file_extension == '.html':
                with open(source, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
            elif file_extension == '.pdf':
                try:
                    with open(source, 'rb') as f:
                        reader = PdfReader(f)
                        pdf_text = []
                        for page in reader.pages:
                            pdf_text.append(page.extract_text())
                        text = "\n".join(pdf_text)
                except Exception as e:
                    print(f"Error reading PDF file {source}: {e}")
                    return None
            else:
                print(f"Error: Unsupported file type '{file_extension}'. Please use JSON, HTML or PDF.")
                return None

        except Exception as e:
            print(f"Error processing file {source}: {e}")
            return None

    # Basic text cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple recursive-style chunking
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        
    print(f"Successfully created {len(chunks)} chunks.")
    return chunks

# --- Module 2: Embedding and Indexing ---

def get_gemini_embedding_function():
    """
    Initializes and returns the Gemini embedding function for ChromaDB.
    """
    if not CHROMADB_AVAILABLE or embedding_functions is None:
        raise ImportError(
            "chromadb (and its embedding utilities) is not available in this environment. "
            "Install chromadb in a clean virtualenv (`pip install chromadb`) or run the FAISS-based pipeline instead."
        )

    return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY,
        model_name=GEMINI_EMBEDDING_MODEL
    )

def create_or_load_vector_store(source, chunks):
    """
    Creates a new ChromaDB vector store or loads an existing one.
    A unique collection name is generated based on the source's content hash.
    Args:
        source (str): Path to the source file or URL.
        chunks (list[str]): List of text chunks from the source.
    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    # Generate a unique and deterministic collection name from the source (file path or URL)
    source_hash = hashlib.md5(source.encode()).hexdigest()
    collection_name = f"{COLLECTION_NAME_PREFIX}{source_hash}"

    if not CHROMADB_AVAILABLE:
        print("chromadb is not installed in this environment. To enable the ChromaDB vector store, install chromadb in a clean venv: \n"
              "  python3 -m venv .venv_chromadb && source .venv_chromadb/bin/activate && pip install --upgrade pip && pip install chromadb\n"
              "Or use the FAISS-based flow (see rag.py) which avoids chromadb's heavy deps.")
        return None

    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = get_gemini_embedding_function()

    # Check if the collection already exists
    try:
        collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
        print(f"Loaded existing collection '{collection_name}' from disk.")
        return collection
    except chromadb.errors.NotFoundError:
        print(f"Collection '{collection_name}' not found. Creating a new one...")
        
    if not chunks:
        print("Error: No chunks provided to create a new collection.")
        return None

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"} # Specifies the distance metric
    )
    
    # Add documents to the collection in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_ids = [f"chunk_{i+j}" for j in range(len(batch_chunks))]
        
        print(f"Adding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} to collection...")
        collection.add(
            documents=batch_chunks,
            ids=batch_ids
        )
    
    print(f"Successfully created and indexed collection '{collection_name}'.")
    return collection

def delete_collection(source):
    """
    Deletes the ChromaDB collection associated with a given source.
    Args:
        source (str): The source (file path or URL) of the collection to delete.
    """
    if not CHROMADB_AVAILABLE:
        print("chromadb is not available. Cannot delete collection.")
        return

    try:
        source_hash = hashlib.md5(source.encode()).hexdigest()
        collection_name = f"{COLLECTION_NAME_PREFIX}{source_hash}"

        client = chromadb.PersistentClient(path=DB_PATH)
        client.delete_collection(name=collection_name)
        print(f"Successfully deleted collection for source: {source}")
    except Exception as e:
        print(f"Error deleting collection for source {source}: {e}")


# --- Module 3: Retrieval and Generation ---

def retrieve_relevant_passages(query, collection, top_k=5):
    """
    Retrieves the top_k most relevant passages from the vector store.
    Args:
        query (str): The user's query.
        collection (chromadb.Collection): The ChromaDB collection.
        top_k (int): The number of passages to retrieve.
    Returns:
        list[str]: A list of the most relevant text passages.
    """
    print(f"Retrieving top {top_k} passages for query: '{query}'...")
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results['documents'][0]

def build_prompt(query, context_passages):
    """
    Constructs the final prompt for the LLM with the retrieved context.
    """
    # This prompt template is crucial for guiding the LLM.
    # It instructs the model to use ONLY the provided context and to be honest if the answer is not found.
    prompt_template = """
    Answer the question based only on the following context:

    {context}

    ---
    Question: {question}
    Answer:
    """
    
    context_str = "\n\n".join(context_passages)
    prompt = prompt_template.format(context=context_str, question=query)
    return prompt

def generate_response(prompt):
    """
    Generates a response from the Gemini model based on the augmented prompt.
    """
    print("Generating response from Gemini...")
    model = genai.GenerativeModel(GEMINI_GENERATIVE_MODEL)
    response = model.generate_content(prompt)
    return response.text

# --- Module 4: Main Execution Block ---

SOURCES_FILE = 'document_sources.json'

def load_sources():
    """Loads the list of document sources from the JSON file."""
    if not os.path.exists(SOURCES_FILE):
        return []
    try:
        with open(SOURCES_FILE, 'r') as f:
            # handle case where file is empty
            content = f.read()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_sources(sources):
    """Saves the list of document sources to the JSON file."""
    with open(SOURCES_FILE, 'w') as f:
        json.dump(sources, f, indent=4)

def main():
    """
    Orchestrates the entire RAG pipeline with a multi-document management system.
    """
    # Load previously saved sources and initialize their vector stores
    saved_sources = load_sources()
    document_stores = {}
    if saved_sources:
        print("Loading previously added documents...")
        for source in saved_sources:
            # We pass `chunks=None` because we expect the collection to exist.
            # If it doesn't, create_or_load_vector_store will handle it gracefully.
            vector_store = create_or_load_vector_store(source, chunks=None)
            if vector_store:
                document_stores[source] = vector_store
        print("---")


    while True:
        print("\n--- Main Menu ---")
        print("1. Add a new document")
        print("2. Ask questions about a document")
        print("3. List loaded documents")
        print("4. Remove a document")
        print("5. Exit")
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            # Add a new document
            source = input("Enter the path to your JSON/HTML/PDF file or a URL to scrape: ")
            if source in document_stores:
                print(f"Document '{source}' is already loaded.")
                continue
            
            chunks = load_and_chunk_data(source)
            if not chunks:
                continue  # Continue to main menu if loading fails

            vector_store = create_or_load_vector_store(source, chunks)
            if vector_store:
                document_stores[source] = vector_store
                # Save the updated list of sources for persistence
                save_sources(list(document_stores.keys()))
                print(f"\nSuccessfully loaded and indexed '{source}'.")

        elif choice == '2':
            # Ask questions about a document
            if not document_stores:
                print("\nNo documents loaded yet. Please add a document first.")
                continue

            print("\n--- Select a Document to Query ---")
            sources = list(document_stores.keys())
            for i, src in enumerate(sources):
                print(f"{i + 1}. {src}")

            try:
                doc_choice = int(input(f"Enter your choice (1-{len(sources)}): ")) - 1
                if not 0 <= doc_choice < len(sources):
                    print("Invalid choice. Please try again.")
                    continue

                selected_source = sources[doc_choice]
                vector_store = document_stores[selected_source]

                print(f"\n--- Querying '{selected_source}' ---")
                print("Type 'back' to return to the main menu.")

                while True:
                    user_query = input("\nYour Question: ")
                    if user_query.lower() == 'back':
                        break

                    context = retrieve_relevant_passages(user_query, vector_store)
                    prompt = build_prompt(user_query, context)
                    answer = generate_response(prompt)

                    print("\n--- Generated Answer ---")
                    print(answer)
                    print("------------------------")

            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == '3':
            # List loaded documents
            if not document_stores:
                print("\nNo documents loaded yet.")
            else:
                print("\n--- Loaded Documents ---")
                for i, source in enumerate(document_stores.keys()):
                    print(f"{i + 1}. {source}")
        
        elif choice == '4':
            # Remove a document
            if not document_stores:
                print("\nNo documents loaded yet.")
                continue

            print("\n--- Select a Document to Remove ---")
            sources = list(document_stores.keys())
            for i, src in enumerate(sources):
                print(f"{i + 1}. {src}")

            try:
                doc_choice = int(input(f"Enter your choice (1-{len(sources)}): ")) - 1
                if not 0 <= doc_choice < len(sources):
                    print("Invalid choice. Please try again.")
                    continue

                selected_source = sources[doc_choice]

                # Delete the collection from ChromaDB
                delete_collection(selected_source)

                # Remove from in-memory store
                del document_stores[selected_source]

                # Update the sources file
                save_sources(list(document_stores.keys()))

                print(f"Successfully removed document: {selected_source}")

            except ValueError:
                print("Invalid input. Please enter a number.")

        elif choice == '5':
            print("Exiting RAG system. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()