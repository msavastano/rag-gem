import os
import re
import hashlib
import google.generativeai as genai
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
GEMINI_GENERATIVE_MODEL = "gemini-1.5-flash"
DB_PATH = "./chroma_db"
COLLECTION_NAME_PREFIX = "rag_collection_"

# --- Module 1: Data Ingestion ---

def load_and_chunk_pdf(file_path, chunk_size=1000, chunk_overlap=100):
    """
    Loads a PDF, extracts text, and splits it into overlapping chunks.
    This uses a simple recursive-style splitting logic.
    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int): The maximum size of each chunk in characters.
        chunk_overlap (int): The number of characters to overlap between chunks.
    Returns:
        list[str]: A list of text chunks.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    print(f"Loading and chunking PDF: {file_path}...")
    reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in reader.pages)
    
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

def create_or_load_vector_store(file_path, chunks):
    """
    Creates a new ChromaDB vector store or loads an existing one.
    A unique collection name is generated based on the file's content hash.
    Args:
        file_path (str): Path to the source PDF file.
        chunks (list[str]): List of text chunks from the PDF.
    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    # Generate a unique and deterministic collection name from the file path
    # This helps in loading the correct pre-indexed collection for a given file
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    collection_name = f"{COLLECTION_NAME_PREFIX}{file_hash}"

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
    except ValueError:
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
    return results['documents']

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

def main():
    """
    Orchestrates the entire RAG pipeline.
    """
    # 1. Get PDF file path from user
    pdf_file_path = input("Enter the path to your PDF file: ")
    
    # 2. Ingest and Process the Document
    chunks = load_and_chunk_pdf(pdf_file_path)
    if not chunks:
        return # Exit if loading failed
        
    # 3. Create or Load the Vector Store
    vector_store = create_or_load_vector_store(pdf_file_path, chunks)
    if not vector_store:
        return # Exit if vector store creation failed

    # 4. Interactive Q&A Loop
    print("\n--- RAG System Ready. Ask questions about your document. ---")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting RAG system. Goodbye!")
            break
            
        # 5. Retrieve
        context = retrieve_relevant_passages(user_query, vector_store)
        
        # 6. Augment
        prompt = build_prompt(user_query, context)
        
        # 7. Generate
        answer = generate_response(prompt)
        
        print("\n--- Generated Answer ---")
        print(answer)
        print("------------------------")

if __name__ == "__main__":
    main()