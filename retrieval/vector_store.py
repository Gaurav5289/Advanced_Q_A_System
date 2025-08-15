from langchain_pinecone import PineconeVectorStore
from core.llm_provider import embedding_model
from core.config import settings

def get_vector_store():
    """Initializes and returns the Pinecone vector store."""
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embedding_model
    )
    return vector_store 