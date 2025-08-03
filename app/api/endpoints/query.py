# app/api/endpoints/query.py
import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException

from app.models.query import QueryRequest, QueryResponse # Make sure you have these Pydantic models

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/hackrx/run", response_model=QueryResponse, tags=["Submissions"])
async def run_submission(request_body: QueryRequest, request: Request):
    """
    Processes a document URL and a list of questions.
    It uses a cached vector store to avoid reprocessing the same document,
    significantly improving performance.
    """
    # Access the pipeline and cache from the application state
    rag_pipeline = request.app.state.rag_pipeline
    vector_store_cache = request.app.state.vector_store_cache
    
    doc_url = request_body.documents
    if not doc_url:
        raise HTTPException(status_code=400, detail="No document URL provided.")

    try:
        # --- CACHING LOGIC ---
        # Check if the vector store for this URL is already in our cache
        if doc_url not in vector_store_cache:
            logger.info(f"Document not in cache. Processing and caching: {doc_url}")
            # If not, download, process, and store it in the cache
            local_pdf_path = await rag_pipeline.download_and_get_path(doc_url)
            vector_store = await rag_pipeline.create_vector_store([local_pdf_path])
            vector_store_cache[doc_url] = vector_store
        else:
            logger.info(f"Using cached vector store for document: {doc_url}")
            vector_store = vector_store_cache[doc_url]
        # --- END CACHING LOGIC ---

        # Process all questions in parallel for maximum speed
        tasks = [
            rag_pipeline.answer_question(question, vector_store)
            for question in request_body.questions
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        answers = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"An error occurred while answering a question: {res}")
                answers.append("Error: Could not process this question due to an internal error.")
            else:
                # Assuming answer_question returns the answer string directly
                answers.append(res)
                
        return QueryResponse(answers=answers)

    except Exception as e:
        logger.error(f"A critical error occurred in the query endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")