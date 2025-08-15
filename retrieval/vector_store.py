# retrieval/vector_store.py
from langchain_community.vectorstores import Chroma
from core.llm_provider import embedding_model

CHROMA_PATH = "chroma_db"

def get_vector_store():
    """Initializes and returns the local Chroma vector store."""
    print("INFO: Loading local ChromaDB vector store.")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_model
    )
    return vector_store