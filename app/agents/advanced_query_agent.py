# app/agents/advanced_query_agent.py
import logging
import asyncio
from typing import List, Dict, Any, Tuple
from app.models.query import QueryRequest, QueryResponse, DetailedAnswer, SourceClause
from app.core.rag_pipeline import HybridRAGPipeline, OptimizedVectorStore

logger = logging.getLogger(__name__)

class AdvancedQueryAgent:
    """
    An advanced agent that can create multi-step plans to answer complex queries.
    """

    def __init__(self, rag_pipeline: HybridRAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.vector_store: OptimizedVectorStore = None
        self.history = {}  # To store results of sub-queries

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Main entry point for the agent."""
        logger.info(f"Advanced Agent received request for document: {request.documents}")
        
        # Step 1: Ensure the document is processed and we have a vector store
        self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)

        # Step 2: Process questions concurrently
        tasks = [self.answer_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)

        return QueryResponse(answers=answers)

    async def answer_question(self, question: str) -> str:
        """
        Plans and executes the steps needed to answer a single question.
        """
        logger.info(f"Agent is planning an answer for: '{question}'")

        # A simple planning mechanism:
        if "compare" in question.lower() and "and" in question.lower():
            # This is a complex comparison question, break it down
            try:
                # Example: "Compare the pre-hospitalization period and the waiting period"
                parts = question.split("compare")[1].split("and")
                topic1_query = f"What is {parts[0].strip()}?"
                topic2_query = f"What is {parts[1].strip()}?"

                logger.info(f"Decomposed into sub-problems: ['{topic1_query}', '{topic2_query}']")

                # Execute sub-queries in parallel
                sub_answers = await asyncio.gather(
                    self.execute_sub_query(topic1_query),
                    self.execute_sub_query(topic2_query)
                )

                # Synthesize the final answer
                final_answer = await self.synthesize_answer(question, sub_answers)
                return final_answer
            except Exception as e:
                logger.warning(f"Failed to decompose complex query, using direct approach. Error: {e}")
                # Fallback to direct approach if decomposition fails
                return await self.execute_sub_query(question)
        else:
            # This is a direct question, answer it in one step
            return await self.execute_sub_query(question)

    async def execute_sub_query(self, sub_query: str) -> str:
        """
        Executes a single, direct query against the RAG pipeline.
        This is our primary "tool."
        """
        logger.info(f"Executing tool for sub-query: '{sub_query}'")
        # Use the existing answer generation from the pipeline
        answer = await self.rag_pipeline.answer_question(sub_query, self.vector_store)
        # Store result in history for potential future use
        self.history[sub_query] = answer
        return answer

    async def synthesize_answer(self, original_question: str, sub_answers: List[str]) -> str:
        """

        Uses an LLM to combine the answers of sub-queries into a cohesive final answer.
        """
        logger.info("Synthesizing final answer from sub-problems.")
        
        context = "\n\n".join(
            f"Regarding the question '{k}', the information found is: {v}"
            for k, v in self.history.items() if v in sub_answers
        )

        synthesis_prompt = f"""
        You are a helpful AI assistant. You have been asked a complex question and have gathered the following pieces of information by asking simpler questions.
        
        Original Complex Question: {original_question}

        Gathered Information:
        ---
        {context}
        ---

        Based *only* on the gathered information, provide a clear and direct answer to the original complex question.
        """
        
        # This uses the same underlying LLM call as the main pipeline
        # but with a different, specific prompt.
        model = self.rag_pipeline.llm_precise
        response = await model.generate_content_async(synthesis_prompt)

        return response.text.strip()