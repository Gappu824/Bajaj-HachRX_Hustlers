# app/models/query.py

from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    """
    Defines the structure for an incoming API request, matching the problem statement.
    """
    # --- THIS IS THE FIX ---
    # Change this from List[str] to str to match the API documentation.
    documents: str = Field(..., description="The URL of the single document to be processed.")
    # --- END FIX ---
    questions: List[str] = Field(..., description="A list of questions to be answered about the document.")

class QueryResponse(BaseModel):
    """
    Defines the structure for the API's response.
    """
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions asked.")