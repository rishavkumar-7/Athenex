# main.py
import logging
import os
import shutil
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from llama_index.core.prompts import PromptTemplate

# --- Assuming these imports reflect your actual files ---
from indexes import initialize_index, insert_into_index, query_index, setup_embedding
from query_model import model_call
# --- End Imports ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Limit in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
# --- MODIFIED PATH ---
# Using the absolute path provided by the user with a raw string (r"...")
PYQS_DIR =  os.path.join(BASE_DIR, "dataset\pyqs")
# --- END MODIFIED PATH ---

# Ensure directories exist (will only create documents dir relative to script)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
# We cannot reliably create the absolute PYQS_DIR here, assume it exists.
# os.makedirs(PYQS_DIR, exist_ok=True) # Commented out for absolute path

# Global config (simulating in-memory session config)
user_config = {
    "provider": None,
    "model_name": None,
    "api_key": None,
    "initialized": False # Track if base embedding/index is ready
}

# Model for config input
class ConfigRequest(BaseModel):
    provider: str
    modelName: str
    apiKey: str | None = None

@app.post("/set_config")
async def set_config(config: ConfigRequest):
    """
    Receives model configuration from frontend and stores it.
    Initializes embedding model and general document indices.
    """
    logger.info(f"Setting configuration: Provider={config.provider}, Model={config.modelName}")
    user_config["provider"] = config.provider
    user_config["model_name"] = config.modelName
    user_config["api_key"] = config.apiKey if config.apiKey else "N/A"
    user_config["initialized"] = False

    try:
        logger.info("Setting up embedding model...")
        setup_embedding()
        logger.info("Initializing base document indices from ./documents...")
        initialize_index()
        user_config["initialized"] = True
        logger.info("Base model configuration and initial index setup complete.")
        return {"message": f"Model set to {config.modelName}. Base setup complete."}
    except Exception as e:
        logger.error(f"Error during base initialization: {e}", exc_info=True)
        user_config["initialized"] = False
        raise HTTPException(status_code=500, detail=f"Configuration set, but failed to initialize backend: {e}")

# --- PYQS Functionality ---

@app.get("/list_pyqs_folders")
async def list_pyqs_folders():
    """Lists subdirectories in the PYQS_DIR."""
    logger.info(f"Listing folders in PYQS directory: {PYQS_DIR}")
    # Check if the absolute path exists
    if not os.path.exists(PYQS_DIR) or not os.path.isdir(PYQS_DIR):
        logger.error(f"Configured PYQS directory not found or not accessible: {PYQS_DIR}")
        raise HTTPException(status_code=404, detail=f"PYQS directory not found on server at configured path: {PYQS_DIR}")
    try:
        folders = [d for d in os.listdir(PYQS_DIR) if os.path.isdir(os.path.join(PYQS_DIR, d))]
        logger.info(f"Found PYQS folders: {folders}")
        return {"folders": folders}
    except Exception as e:
        logger.error(f"Error listing PYQS folders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading PYQS folders: {e}")

@app.post("/load_pyqs_folder_to_docs")
async def load_pyqs_folder_to_docs(folder_name: str = Form(...)):
    """Copies files from selected PYQS folder to DOCUMENTS_DIR and re-initializes index."""
    logger.info(f"Request to load PYQS folder '{folder_name}' into documents.")

    if not user_config["initialized"]:
        raise HTTPException(status_code=400, detail="Base model configuration must be completed first.")

    # Use the absolute path for source
    source_folder_path = os.path.join(PYQS_DIR, folder_name)
    if not os.path.isdir(source_folder_path):
        logger.error(f"Source PYQS folder '{folder_name}' not found at {source_folder_path}.")
        raise HTTPException(status_code=404, detail=f"PYQS folder '{folder_name}' not found.")

    # Marker file check remains in the relative DOCUMENTS_DIR
    marker_file_name = f".pyqs_loaded_{folder_name.replace(' ', '_')}"
    marker_file_path = os.path.join(DOCUMENTS_DIR, marker_file_name)

    if os.path.exists(marker_file_path):
        logger.info(f"PYQS folder '{folder_name}' content appears to be already loaded (marker found). Skipping copy.")
        try:
            logger.info(f"Re-initializing index based on current documents (including previously loaded {folder_name})...")
            initialize_index() # Re-read ./documents
            logger.info(f"Index re-initialized successfully after confirming '{folder_name}' was loaded.")
            return {"message": f"PYQS folder '{folder_name}' is already loaded. Index refreshed."}
        except Exception as e:
            logger.error(f"Error re-initializing index after skipping copy for '{folder_name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Folder already loaded, but failed to refresh index: {e}")

    logger.info(f"Copying files from '{source_folder_path}' to '{DOCUMENTS_DIR}'...")
    try:
        # Clear previous marker files
        for item in os.listdir(DOCUMENTS_DIR):
            if item.startswith(".pyqs_loaded_"):
                try: os.remove(os.path.join(DOCUMENTS_DIR, item))
                except OSError: logger.warning(f"Could not remove old marker file: {item}")

        # Copy files
        copied_count = 0
        for item in os.listdir(source_folder_path):
            source_item_path = os.path.join(source_folder_path, item)
            dest_item_path = os.path.join(DOCUMENTS_DIR, item) # Destination is still relative ./documents
            if os.path.isfile(source_item_path):
                shutil.copy2(source_item_path, dest_item_path)
                logger.debug(f"Copied '{item}' to documents directory.")
                copied_count += 1

        if copied_count == 0:
             logger.warning(f"No files found to copy in PYQS folder: '{folder_name}'")
             with open(marker_file_path, 'w') as f: f.write('')
             return {"message": f"No files found in PYQS folder '{folder_name}'. Nothing copied."}

        logger.info(f"Copied {copied_count} files. Re-initializing index based on updated documents...")
        initialize_index()
        logger.info("Index re-initialized successfully.")
        with open(marker_file_path, 'w') as f: f.write('') # Create marker

        return {"message": f"Successfully loaded {copied_count} file(s) from '{folder_name}' and updated index."}

    except Exception as e:
        logger.error(f"Error copying files from '{folder_name}' or re-initializing index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load PYQS folder '{folder_name}': {e}")


# --- General Document Upload ---
@app.post("/uploadFile")
async def upload_file(file: UploadFile = File(...), filename_as_doc_id: str = Form(None)):
    """Upload a file and add it to the current index based on ./documents."""
    # (This function remains the same as the previous version)
    if not user_config["initialized"]:
        raise HTTPException(status_code=400, detail="Model configuration must be completed before uploading files.")
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    file_path = os.path.join(DOCUMENTS_DIR, os.path.basename(file.filename))
    logger.info(f"Receiving uploaded file: {file.filename}")
    try:
        with open(file_path, "wb") as f: shutil.copyfileobj(file.file, f)
        logger.info(f"File saved to: {file_path}")
        logger.info("Inserting uploaded file into index...")
        insert_into_index(file_path, doc_id=filename_as_doc_id)
        logger.info("File insertion complete.")
        return {"message": f"File '{file.filename}' uploaded and added to the index successfully!"}
    except Exception as e:
        logger.error(f"Error during file upload or indexing: {e}", exc_info=True)
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except OSError: logger.error(f"Could not remove partial file: {file_path}")
        raise HTTPException(status_code=500, detail=f"Upload or indexing failed: {str(e)}")

# --- Query Endpoint ---
@app.post("/query")
async def query_endpoint(request: Request):
    """Query the RAG system. Always uses the index based on current ./documents."""
    # (This function remains the same as the previous version)
    if not user_config["initialized"]:
        raise HTTPException(status_code=400, detail="Model configuration must be completed before querying.")
    try: data = await request.json()
    except Exception: raise HTTPException(status_code=400, detail="Invalid request body. Expected JSON.")
    query_text = data.get("text")
    use_summary = data.get("use_summary", False)
    if not query_text: raise HTTPException(status_code=400, detail="Query text is required")
    logger.info(f"Received query: '{query_text}', Use Summary: {use_summary}")
    try:
        response_text = get_response_from_model(query_text, use_summary)
        logger.info("Response generated successfully.")
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error getting response from model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get response: {e}")

def get_response_from_model(query: str, use_summary: bool = False):
    """Get response using RAG context (from current ./documents index) and LLM."""
    # (This function remains the same as the previous version)
    logger.info(f"Fetching context using query_index (summary={use_summary})...")
    context = query_index(query, use_summary=use_summary)
    logger.debug(f"Retrieved context: {' '.join(context.split()[:30])}...")
    prompt = PromptTemplate(
        "You are an intelligent assistant that answers user queries based on the retrieved context provided below.\n\n"
        "Formatting Instructions:\n"
        "- If your answer contains code, always wrap it in a proper code block using triple backticks (```).\n"
        "- If the code is in a specific language (e.g. Python, JavaScript), mention the language after the opening backticks.\n"
        "- Be concise and helpful. If the context does not have enough information, clearly say so.\n"
        "- Keep general answers easy to read and break long responses into paragraphs if necessary.\n\n"
        "Retrieved Context:\n{context}\n\n"
        "User Question:\n{query}"
    ).format(context=context or "No relevant context found.", query=query)
    logger.info("Calling LLM with context-enhanced prompt...")
    response = model_call(user_config, query=prompt)
    return response

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    # Check if the configured PYQS_DIR exists on startup
    if not os.path.exists(PYQS_DIR) or not os.path.isdir(PYQS_DIR):
         logger.warning(f"Warning: The configured PYQS directory does not exist or is not accessible: {PYQS_DIR}")
         logger.warning("The 'Load PYQS Folders' feature will likely fail.")
    # Initialization happens after /set_config now
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)



















































































































































































# from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn
# import os
# import shutil
# from llama_index.core.prompts import PromptTemplate
# from indexes import initialize_index, insert_into_index, query_index, setup_embedding
# from query_model import model_call

# app = FastAPI()

# # Allow frontend to access backend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Limit in production!
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create directories if they don't exist
# DOCUMENTS_DIR = "./documents"
# os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# # Global config (simulating in-memory session config)
# user_config = {
#     "provider": None,
#     "model_name": None,
#     "api_key": None
# }

# # Model for config input
# class ConfigRequest(BaseModel):
#     provider: str
#     modelName: str
#     apiKey: str | None = None  # Not needed for local

# class QueryRequest(BaseModel):
#     text: str
#     use_summary: bool = False  # Default to vector index

# @app.post("/set_config")
# async def set_config(config: ConfigRequest):
#     """
#     Receives model configuration from frontend and stores it in memory.
#     """
#     user_config["provider"] = config.provider
#     user_config["model_name"] = config.modelName
#     user_config["api_key"] = config.apiKey if config.apiKey else "N/A"

#     # Initialize embedding + index only after config is set
#     setup_embedding()
#     initialize_index()

#     # Add any backend setup logic here (like model loading, etc.)
#     return {"message": f"Model set to {config.modelName} under {config.provider}."}

# @app.post("/uploadFile")
# async def upload_file(file: UploadFile = File(...), filename_as_doc_id: str = Form(None)):
#     """
#     Upload a file and add it to both vector and summary indices
#     """
#     if not file:
#         raise HTTPException(status_code=400, detail="No file uploaded")

#     file_path = os.path.join(DOCUMENTS_DIR, os.path.basename(file.filename))

#     try:
#         # Save the file temporarily
#         with open(file_path, "wb") as f:
#             shutil.copyfileobj(file.file, f)

#         # Add to indices
#         insert_into_index(file_path, doc_id=filename_as_doc_id)
#         return {"message": "File inserted successfully into both vector and summary indices!"}

#     except Exception as e:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# @app.post("/query")
# async def query_endpoint(request: Request):
#     """
#     Query the RAG system with user text
#     """
#     data = await request.json()
#     query_text = data.get("text")
#     use_summary = data.get("use_summary", False)
    
#     if not query_text:
#         return {"response": "Error: Query text is required"}

#     # Log the request
#     print("User Config:", user_config)
#     print(f"Received query: {query_text}")
#     print(f"Using summary index: {use_summary}")

#     # Get response using RAG and LLM
#     # implement loop for chat 
#     response = get_response_from_model(query_text, use_summary)
#     print("response",response)
#     return {"response": response}

# def get_response_from_model(query: str, use_summary: bool = False):
#     """
#     Get response using RAG context and LLM
#     """
#     # Get context from appropriate index
#     context = query_index(query, use_summary=use_summary)
    
#     # Create prompt with context
#     prompt = PromptTemplate(
#         "You are an intelligent assistant that answers user queries based on the retrieved context provided below.\n\n"
#         "Formatting Instructions:\n"
#         "- If your answer contains code, always wrap it in a proper code block using triple backticks (```).\n"
#         "- If the code is in a specific language (e.g. Python, JavaScript), mention the language after the opening backticks.\n"
#         "- Be concise and helpful. If the context does not have enough information, clearly say so.\n"
#         "- Keep general answers easy to read and break long responses into paragraphs if necessary.\n\n"
#         "Retrieved Context:\n{context}\n\n"
#         "User Question:\n{query}"
#     ).format(context=context, query=query)
    
#     # Call LLM with prompt
#     response = model_call(user_config, query=prompt)
#     return response

# if __name__ == "__main__":
#     print("Initializing embedding model and indices...")
#     setup_embedding()
#     initialize_index()

#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
