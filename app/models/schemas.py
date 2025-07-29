# app/models/schemas.py
from typing import List
from pydantic import BaseModel, HttpUrl

# Model for the incoming POST request body
class QueryRequest(BaseModel):
    documents: List[HttpUrl]
    questions: List[str]

# Model for the outgoing JSON response
class QueryResponse(BaseModel):
    answers: List[str]