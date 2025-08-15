from langchain.retrievers.multi_query import MultiQueryRetriever
from core.llm_provider import llm

def create_multi_query_retriever(base_retriever):
    """
    Creates a multi-query retriever that expands a user's query.
    """
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    return retriever