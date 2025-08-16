from fastapi import APIRouter, HTTPException
from .schemas import AskRequest, AskResponse
from orchestrator.agent import ask_question
from core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Accepts a question and returns an answer from the RAG system.
    """
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
            
        logger.info(f"Received question: {question}")
        answer = ask_question(question)
        logger.info("Successfully generated an answer.")
        
        return AskResponse(answer=answer)
        
    except Exception as e:
        logger.error(f"An error occurred during /ask request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")