from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(docs: list):
    """Chunks documents using a recursive character text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    
    print("Chunking documents...")
    chunked_documents = text_splitter.split_documents(docs)
    print(f"Created {len(chunked_documents)} chunks.")
    return chunked_documents