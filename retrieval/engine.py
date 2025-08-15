from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from .vector_store import get_vector_store
from .query_transformer import create_multi_query_retriever

def create_retrieval_engine(top_k=10, rerank_top_n=3):
    """
    Creates an advanced retrieval engine with query transformation and re-ranking.
    """
    vector_store = get_vector_store()
    
    # 1. Create the base retriever from the vector store
    base_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    
    # 2. Apply the MultiQueryRetriever for query expansion
    multi_query_retriever = create_multi_query_retriever(base_retriever)

    # 3. Initialize the cross-encoder model for re-ranking
    cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 4. Create the re-ranker compressor
    compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=rerank_top_n)
    
    # 5. Create the final Contextual Compression Retriever
    #    The re-ranker will run on the results from the multi-query retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=multi_query_retriever # Use the enhanced retriever
    )
    
    return compression_retriever