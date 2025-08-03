# app/api/endpoints/query.py
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException

from app.models.query import QueryRequest, QueryResponse # Make sure you have these Pydantic models

logger = logging.getLogger(__name__)
router = APIRouter()

# In app/api/endpoints/query.py

@router.post("/hackrx/run", response_model=QueryResponse, tags=["Submissions"])
async def run_submission(request_body: QueryRequest, request: Request):
    """
    Processes a document URL and a list of questions by calling the RAG pipeline.
    The pipeline itself is responsible for all caching and processing logic.
    """
    # 1. Access the RAG pipeline from the application state.
    rag_pipeline = request.app.state.rag_pipeline

    # 2. Validate and sanitize the incoming document URL. This part is correct!
    if not request_body.documents or not isinstance(request_body.documents, list):
        raise HTTPException(status_code=400, detail="Request body must include a non-empty 'documents' list.")
    
    url_from_list = request_body.documents[0]
    doc_url = url_from_list.strip()
    questions = request_body.questions

    try:
        # 3. Delegate the entire process to the pipeline with a single, clean call.
        # The 'process_query' method in your pipeline should handle everything:
        # checking the cache, downloading if needed, creating the vector store,
        # and answering the questions.
        answers = await rag_pipeline.process_query(doc_url, questions)
        
        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"A critical error occurred in the query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")