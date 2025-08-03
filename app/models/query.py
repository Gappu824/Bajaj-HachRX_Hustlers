# app/models/query.py - Enhanced with explainability

from pydantic import BaseModel, Field
from typing import List, Optional

class SourceClause(BaseModel):
    """Represents a source clause used in the answer"""
    text: str = Field(..., description="The actual text of the clause")
    confidence_score: float = Field(..., description="Confidence score for this clause")
    page_number: Optional[int] = Field(None, description="Page number where clause was found")
    section: Optional[str] = Field(None, descriptiona="Document section")

class DetailedAnswer(BaseModel):
    """Enhanced answer with explainability"""
    answer: str = Field(..., description="The main answer to the question")
    confidence: float = Field(..., description="Overall confidence in the answer (0-1)")
    source_clauses: List[SourceClause] = Field(..., description="Source clauses supporting the answer")
    reasoning: str = Field(..., description="Explanation of how the answer was derived")
    coverage_decision: Optional[str] = Field(None, description="Clear coverage decision (Covered/Not Covered/Conditional)")

class QueryRequest(BaseModel):
    """
    Defines the structure for an incoming API request, matching the problem statement.
    """
    documents: str = Field(..., description="The URL of the single document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered about the document.")

class QueryResponse(BaseModel):
    """
    Enhanced response structure with explainability
    """
    answers: List[str] = Field(..., description="Simple answers for backward compatibility")
    detailed_answers: Optional[List[DetailedAnswer]] = Field(None, description="Detailed answers with explainability")
    processing_time: Optional[float] = Field(None, description="Time taken to process the request")