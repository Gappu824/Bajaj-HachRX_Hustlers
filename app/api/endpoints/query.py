# app/api/endpoints/query.py - Enhanced endpoint
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException, Query
from app.models.query import QueryRequest, QueryResponse


logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/hackrx/run", response_model=QueryResponse, tags=["Submissions"])
async def run_submission(
    request_body: QueryRequest, 
    request: Request,
    explainable: bool = Query(False, description="Include detailed explanations and reasoning")
):
    """
    Processes a document URL and a list of questions.
    Set explainable=true for detailed answers with reasoning and clause traceability.
    """
    rag_pipeline = request.app.state.rag_pipeline
    doc_url = request_body.documents.strip()
    questions = request_body.questions

    try:
        if explainable:
            # Use enhanced processing with explainability
            simple_answers, detailed_answers, processing_time = await rag_pipeline.process_query_with_explainability(doc_url, questions)
            return QueryResponse(
                answers=simple_answers,
                detailed_answers=detailed_answers,
                processing_time=processing_time
            )
        else:
            # Use existing simple processing for backward compatibility
            answers = await rag_pipeline.process_query(doc_url, questions)
            return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"A critical error occurred in the query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@router.post("/hackrx/run/detailed", response_model=QueryResponse, tags=["Submissions"])
async def run_detailed_submission(request_body: QueryRequest, request: Request):
    """
    Enhanced endpoint that always returns detailed explanations.
    Provides full explainability, decision rationale, and clause traceability.
    """
    rag_pipeline = request.app.state.rag_pipeline
    doc_url = request_body.documents.strip()
    questions = request_body.questions

    try:
        simple_answers, detailed_answers, processing_time = await rag_pipeline.process_query_with_explainability(doc_url, questions)
        return QueryResponse(
            answers=simple_answers,
            detailed_answers=detailed_answers,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"A critical error occurred in the detailed query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")