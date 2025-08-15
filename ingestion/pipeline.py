# ingestion/pipeline.py
from langchain_community.vectorstores import Chroma
from langchain.schema import Document as LangChainDocument
from .load import load_documents
from .chunk import chunk_documents
from core.llm_provider import embedding_model

CHROMA_PATH = "chroma_db"

def run_ingestion_pipeline():
    llama_parse_docs = load_documents()
    if not llama_parse_docs:
        print("No documents were loaded. Exiting.")
        return
        
    docs_to_chunk = [
        LangChainDocument(page_content=doc.text, metadata=doc.metadata) 
        for doc in llama_parse_docs
    ]
    chunked_docs = chunk_documents(docs_to_chunk)
    
    print(f"Creating/updating local vector store at: {CHROMA_PATH}")
    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    print("\n✅ Ingestion to local ChromaDB complete!")