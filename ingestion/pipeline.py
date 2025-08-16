from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document as LangChainDocument
from .load import load_documents
from .chunk import chunk_documents
from core.llm_provider import embedding_model
from core.config import settings

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
    
    print(f"Creating embeddings and loading to Pinecone index '{settings.PINECONE_INDEX_NAME}'...")
    PineconeVectorStore.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        index_name=settings.PINECONE_INDEX_NAME
    )
    
    print("\n✅ Ingestion to Pinecone complete!")