# app/api/endpoints/query.py
import logging
from fastapi import APIRouter, Depends, HTTPException, Security, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.models.schemas import QueryRequest, QueryResponse
from app.core.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()
bearer_scheme = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """Validates the static bearer token from the problem statement."""
    if credentials.scheme != "Bearer" or credentials.credentials != settings.BEARER_TOKEN:
        logger.warning("Invalid API token received.")
        raise HTTPException(status_code=403, detail="Invalid or missing API token")
    return credentials

@router.post("/hackrx/run", response_model=QueryResponse, tags=["Query"])
async def run_query(
    fastapi_req: Request,
    request_data: QueryRequest,
    token: str = Depends(validate_token)
):
    """
    Main API endpoint to process documents and answer questions.
    """
    if not request_data.documents:
        raise HTTPException(status_code=400, detail="No document URL provided.")
    
    # This solution handles the first document URL, as per the sample request.
    # NEW, MORE ROBUST CODE
    document_url = request_data.documents.strip()
    logger.info(f"Received request for document: {document_url} with {len(request_data.questions)} questions.")

    try:
        # Get the initialized RAG pipeline instance from the app's state
        rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
        answers = await rag_pipeline.process_query(document_url, request_data.questions)
        return QueryResponse(answers=answers)
    except ValueError as e:
        # Handle known processing errors (e.g., empty PDF)
        logger.error(f"Value Error processing request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch-all for unexpected server errors
        logger.critical(f"An unexpected server error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")