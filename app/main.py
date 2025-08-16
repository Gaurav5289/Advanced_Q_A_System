from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(
    title="Advanced Q&A System API",
    description="An API for asking questions to an advanced RAG system.",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api")

@app.get("/", tags=["Root"])
async def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Advanced Q&A System API!"}