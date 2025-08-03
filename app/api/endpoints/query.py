# app/api/endpoints/query.py
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException
from app.models.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/hackrx/run", response_model=QueryResponse, tags=["Submissions"])
async def run_submission(request_body: QueryRequest, request: Request):
    """
    Processes a document URL and a list of questions.
    """
    rag_pipeline = request.app.state.rag_pipeline

    # --- SIMPLIFIED LOGIC ---
    # 'request_body.documents' is now a string, so we can use it directly.
    # The .strip() is still a good idea for robustness.
    doc_url = request_body.documents.strip()
    questions = request_body.questions
    # --- END SIMPLIFIED LOGIC ---

    try:
        # Pass the clean string and questions list to the pipeline.
        answers = await rag_pipeline.process_query(doc_url, questions)
        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"A critical error occurred in the query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")