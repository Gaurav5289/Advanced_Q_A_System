from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from retrieval.engine import create_retrieval_engine
from core.llm_provider import llm # Import the shared LLM instance

def create_qa_chain():
    """Creates the main Question-Answering chain using Gemini."""
    
    # Get our advanced retriever
    retriever = create_retrieval_engine()

    # Define the prompt template
    prompt_template = """
    You are an expert Q&A assistant. Your tone should be helpful and professional.
    Use the following pieces of retrieved context to answer the question.
    If the context does not contain the answer, state that you don't have enough information to answer.
    Do not make up an answer. Keep your answer concise.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm # Use the imported shared llm
        | StrOutputParser()
    )
    
    return rag_chain

# Create a single chain instance to be used by the API
qa_chain = create_qa_chain()

def ask_question(question: str):
    """Invokes the QA chain with a user's question."""
    return qa_chain.invoke(question)