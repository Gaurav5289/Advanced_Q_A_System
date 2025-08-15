from langchain.tools import Tool
from retrieval.engine import create_retrieval_engine

def get_retriever_tool():
    """Returns a Tool that wraps the advanced retrieval engine."""
    retriever = create_retrieval_engine()
    
    retriever_tool = Tool(
        name="document_retriever",
        func=retriever.invoke, # LangChain's standard method for running retrievers
        description="Searches and returns relevant information from the knowledge base of documents.",
    )
    return retriever_tool