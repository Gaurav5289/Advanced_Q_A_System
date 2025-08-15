# retrieval/engine.py
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from .vector_store import get_vector_store
from .query_transformer import create_multi_query_retriever

def create_retrieval_engine(top_k=10, rerank_top_n=3):
    """Creates the full, advanced retrieval engine with re-ranking."""
    vector_store = get_vector_store()
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    multi_query_retriever = create_multi_query_retriever(base_retriever)

    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=rerank_top_n)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_query_retriever
    )
    
    return compression_retriever