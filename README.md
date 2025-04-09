
# 📘 Athenex — Multimodal RAG Chatbot

**StudyMate** is an advanced Retrieval-Augmented Generation (RAG) chatbot that supports querying uploaded documents and structured academic datasets (like PYQs). It intelligently selects between **Vector Search** and **Summary Index** using tool-calling with multiple LLM providers (OpenAI, Gemini, local models, etc.).


## 🌟 Key Features

- 🔁 **Dynamic model selection**: OpenAI, Gemini, HuggingFace, Local (Ollama)
- 📄 **Upload and query documents** (`.pdf`, `.txt`, `.docx`, `.md`, etc.)
- 🧠 **Tool calling** for auto-selection between vector or summary index
- 📁 **Load & index academic question folders (PYQS)**
- ⚙️ Built with **FastAPI backend** and a clean **HTML+JS frontend**
- 🎯 LlamaIndex-based document processing

## 🚀 Getting Started

### 🔧 Installation

```bash
# Clone the repo
git clone https://github.com/rishavkumar-7/Athenex.git
cd Athenex

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt
```

### 🏁 Run the App

```bash
uvicorn main:app --reload
```

> Open `index.html` in your browser to interact with the chatbot frontend.

## 🔐 Configuration

- API key is required for OpenAI and Gemini models.
- Local models require Ollama to be running locally with models like `llama3`, `mistral`, etc.
- All documents go inside the `documents/` folder.
- Academic folder loading is done from `dataset/pyqs/`.

## 📂 Folder Structure

```
.
├── main.py               # FastAPI backend
├── query_model.py        # Tool calling logic and LLM routing
├── indexes.py            # Embedding & Index logic (vector & summary)
├── tools.py              # Custom tools for vector/summary querying
├── index.html            # Frontend (clean, dark mode UI)
├── documents/            # Folder for uploaded docs
├── dataset/pyqs/         # Folder structure for PYQS batch loading
├── requirements.txt
├── README.md
```

## 🧠 Tech Stack

- LlamaIndex
- FastAPI
- HuggingFace Embeddings
- Google Generative AI / OpenAI / Ollama
- JavaScript + HTML for frontend
