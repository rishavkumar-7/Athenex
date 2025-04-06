from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from multiprocessing import Lock
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

from llama_index.core.settings import Settings

# Disable default LLM to avoid OpenAI errors until config is set
Settings.llm = None

# --- Config ---
DOCUMENTS_DIR = "./documents"
VECTOR_INDEX_DIR = "./.vector_index"
SUMMARY_INDEX_DIR = "./.summary_index"
vector_index = None
summary_index = None
lock = Lock()

# --- Setup embedding ---
def setup_embedding():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- Load or build indices ---
def initialize_index():
    global vector_index, summary_index

    with lock:
        # --- VECTOR INDEX ---
        if os.path.exists(os.path.join(VECTOR_INDEX_DIR, "docstore.json")):
            print("Loading existing vector index...")
            vector_storage_context = StorageContext.from_defaults(persist_dir=VECTOR_INDEX_DIR)
            vector_index = load_index_from_storage(vector_storage_context)
        else:
            print("Creating new vector index...")
            documents = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()
            vector_index = VectorStoreIndex.from_documents(documents)
            vector_index.storage_context.persist(persist_dir=VECTOR_INDEX_DIR)

        # --- SUMMARY INDEX ---
        if os.path.exists(os.path.join(SUMMARY_INDEX_DIR, "docstore.json")):
            print("Loading existing summary index...")
            summary_storage_context = StorageContext.from_defaults(persist_dir=SUMMARY_INDEX_DIR)
            summary_index = load_index_from_storage(summary_storage_context)
        else:
            print("Creating new summary index...")
            if 'documents' not in locals():
                documents = SimpleDirectoryReader(DOCUMENTS_DIR).load_data()

            summary_index = SummaryIndex.from_documents(documents)
            summary_index.storage_context.persist(persist_dir=SUMMARY_INDEX_DIR)

# --- Insert new doc into both indices ---
def insert_into_index(filepath, doc_id=None):
    global vector_index, summary_index
    
    print(f"Loading document from {filepath}...")
    document = SimpleDirectoryReader(input_files=[filepath]).load_data()[0]
    if doc_id:
        document.doc_id = doc_id

    with lock:
        print(f"Inserting into vector index...")
        vector_index.insert(document)
        vector_index.storage_context.persist()
        
        print(f"Inserting into summary index...")
        summary_index.insert(document)
        summary_index.storage_context.persist(persist_dir=SUMMARY_INDEX_DIR)

# --- Generate full index setup ---
def generate_index():
    """Creates indices with the provided embedding model"""
    setup_embedding()
    initialize_index()

# --- Query vector index ---
def query_vector_index(query_text):
    global vector_index
    query_engine = vector_index.as_query_engine()
    response = query_engine.query(query_text)
    return str(response)

# --- Query summary index ---
def query_summary_index(query_text):
    global summary_index
    query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
    response = query_engine.query(query_text)
    return str(response)

# --- General query function that can use either index ---
def query_index(query_text, use_summary=None):
    """
    Main query function that uses a router to determine which index to query
    unless explicitly specified by use_summary parameter
    
    Args:
        query_text: The query string from the user
        use_summary: If explicitly set to True or False, bypasses the router
                    If None, uses the router to determine which index to use
    
    Returns:
        str: Response from the appropriate index
    """
    # If use_summary is explicitly set, use that value
    # Otherwise, use the router to determine which index to use
    if use_summary is None:
        # List of keywords that suggest the query is looking for a summary
        summary_keywords = [
            "summary", "summarize", "summarization", "overview", 
            "gist", "brief", "synopsis", "recap", "outline",
            "main points", "key points", "tldr", "in short"
        ]
        
        # Check if any summary keywords are in the query
        use_summary = any(keyword in query_text.lower() for keyword in summary_keywords)
        
        # Log which index is being used
        if use_summary:
            print(f"Router: Using summary index for query: '{query_text}'")
        else:
            print(f"Router: Using vector index for query: '{query_text}'")
    
    # Query the appropriate index
    if use_summary:
        return query_summary_index(query_text)
    else:
        return query_vector_index(query_text)