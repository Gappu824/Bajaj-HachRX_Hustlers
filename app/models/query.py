# app/models/query.py

from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    """
    Defines the structure for an incoming API request.
    """
    documents: str = Field(..., description="The URL of the single document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered about the document.")

class QueryResponse(BaseModel):
    """
    Defines the structure for the API's response.
    """
    answers: List[str] = Field(..., description="A list of answers corresponding to the questions asked.")