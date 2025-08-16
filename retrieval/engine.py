from .vector_store import get_vector_store
from .query_transformer import create_multi_query_retriever

def create_retrieval_engine():
    """
    Creates a simplified retrieval engine without the memory-intensive re-ranker
    to fit within free deployment plan limits.
    """
    vector_store = get_vector_store()
    
    # Create the base retriever that searches the vector store
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # Use the multi-query retriever for better search results
    multi_query_retriever = create_multi_query_retriever(base_retriever)
    
    # We will return this retriever directly, skipping the heavy re-ranker
    return multi_query_retriever