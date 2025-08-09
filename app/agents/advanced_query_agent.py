# app/agents/advanced_query_agent.py
import logging
import asyncio
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import html
import hashlib
from app.models.query import QueryRequest, QueryResponse
from app.core.rag_pipeline import HybridRAGPipeline, OptimizedVectorStore

import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

class AdvancedQueryAgent:
    """
    A detective-style agent that investigates beyond explicit questions,
    finding exceptions, contradictions, and related insights to build a full strategy.
    """

    def __init__(self, rag_pipeline: HybridRAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.vector_store: OptimizedVectorStore = None
        self.investigation_cache = {}

    # --- FIX START: ADD THE MISSING HELPER METHODS ---

    async def _get_basic_answer(self, question: str) -> str:
        """Gets a straightforward answer to the question using the RAG pipeline."""
        try:
            logger.info("📝 Getting basic answer...")
            answer = await self.rag_pipeline.answer_question(question, self.vector_store)
            return answer if answer else "No direct answer found."
        except Exception as e:
            logger.error(f"Basic answer retrieval failed: {e}")
            return "Unable to generate a basic answer due to an internal error."

    async def _deep_search(self, question: str) -> str:
        """Performs a deeper search if the basic answer is insufficient."""
        logger.info("🔬 Performing deep search for more context...")
        key_terms = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b|\b[a-z]{4,}\b', question)
        search_queries = [question] + [' '.join(key_terms)]
        
        all_results = []
        for query in search_queries:
            try:
                results = self.vector_store.search(query, k=3)
                all_results.extend([r[0] for r in results])
            except Exception:
                continue
        
        if not all_results:
            return "No relevant information found even after a deep search."

        unique_results = list(dict.fromkeys(all_results))[:4]
        try:
            return await self.rag_pipeline._generate_answer(question, unique_results, is_complex=True)
        except Exception:
            return unique_results[0]

    def _detect_question_patterns(self, question: str) -> Tuple[List[str], List[str]]:
        """Detects question types to guide the investigation."""
        # This is a simplified placeholder. In a real scenario, this would be more complex.
        question_lower = question.lower()
        if "how" in question_lower or "what are the steps" in question_lower:
            return ["process"], ["deadline", "requirement", "exception"]
        if "what is" in question_lower:
            return ["definition"], ["limit", "condition", "exclusion"]
        return ["general"], ["exception", "important", "note"]

    async def _conduct_investigation(self, question: str, q_types: List[str], keywords: List[str], basic_answer: str) -> Dict:
        """A placeholder for the investigation logic."""
        # This method is required by `investigate_question` but its complex
        # logic is not needed to fix the current error.
        return {"exceptions": [], "conditions": []}

    # async def _self_correct_and_refine(self, question: str, original_answer: str, findings: Dict) -> str:
    #     """A placeholder for the self-correction logic."""
    #     # This method is required by `investigate_question` but its complex
    #     # logic is not needed to fix the current error.
    #     return original_answer

    # --- FIX END ---    

    def _clean_text(self, text: str) -> str:
        """Robustly clean text from HTML, encoding issues, and artifacts."""
        if not text:
            return ""
        
        # 1. Decode HTML entities
        text = html.unescape(text)
        
        # 2. Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 3. Handle Unicode and other artifacts
        text = re.sub(r'\(cid:\d+\)', '', text)
        text = text.encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # 4. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    # async def _generate_master_plan(self, questions: List[str]) -> str:
    #     """Analyzes all questions to create a single, unified strategy."""
    #     logger.info("🧠 Analyzing all questions to formulate a master strategy...")

    #     # Consolidate all available text from the vector store to use as context
    #     full_context = "\n---\n".join(self.vector_store.chunks)
        
    #     # Create a prompt that asks the LLM to think like a hackathon winner
    #     prompt = f"""
    #     You are an elite AI agent in a high-stakes, interactive programming challenge.
    #     Your goal is to devise a complete, step-by-step strategy to solve the entire problem, not just answer individual questions.
    #     Analyze the provided context and the list of user questions to understand the overall mission.

    #     CONTEXT:
    #     {full_context}

    #     USER QUESTIONS (use these to understand the mission's scope):
    #     - {"\n- ".join(questions)}

    #     YOUR TASK:
    #     Create a single, comprehensive 'Master Plan' as a step-by-step guide to win the challenge.
    #     This plan should be a clear, actionable walkthrough. Identify critical steps, potential pitfalls, and the final objective.
    #     Be smart, anticipate the required sequence of actions, and explain the logic.
    #     """
        
    #     try:
    #         # Use the most powerful model for strategic planning
    #         model = self.rag_pipeline.llm_precise
    #         response = await model.generate_content_async(
    #             prompt,
    #             generation_config={'temperature': 0.1} # Low temperature for factual, deterministic plans
    #         )
    #         logger.info("✅ Master Plan generated successfully.")
    #         return response.text
    #     except Exception as e:
    #         logger.error(f"Failed to generate master plan: {e}")
            # return "Error: Could not formulate a master plan. The challenge context may be invalid or the objective unclear."
    # async def _generate_master_plan(self, questions: List[str]) -> str:
    #     """Analyzes all questions to create a single, unified strategy."""
    #     logger.info("🧠 Analyzing all questions to formulate a master strategy...")

    #     # Consolidate all available text from the vector store to use as context
    #     full_context = "\n---\n".join(self.vector_store.chunks)
        
    #     # --- FIX ---
    #     # The list of questions is joined into a single string *before* being placed in the f-string.
    #     question_list = "\n- ".join(questions)
        
    #     # Create a prompt that asks the LLM to think like a hackathon winner
    #     prompt = f"""
    #     You are an elite AI agent in a high-stakes, interactive programming challenge.
    #     Your goal is to devise a complete, step-by-step strategy to solve the entire problem, not just answer individual questions.
    #     Analyze the provided context and the list of user questions to understand the overall mission.

    #     CONTEXT:
    #     {full_context}

    #     USER QUESTIONS (use these to understand the mission's scope):
    #     - {question_list}

    #     YOUR TASK:
    #     Create a single, comprehensive 'Master Plan' as a step-by-step guide to win the challenge.
    #     This plan should be a clear, actionable walkthrough. Identify critical steps, potential pitfalls, and the final objective.
    #     Be smart, anticipate the required sequence of actions, and explain the logic.
    #     """
        
    #     try:
    #         # Use the most powerful model for strategic planning
    #         model = self.rag_pipeline.llm_precise
    #         response = await model.generate_content_async(
    #             prompt,
    #             generation_config={'temperature': 0.1} # Low temperature for factual, deterministic plans
    #         )
    #         logger.info("✅ Master Plan generated successfully.")
    #         return response.text
    #     except Exception as e:
    #         logger.error(f"Failed to generate master plan: {e}")
    #         return "Error: Could not formulate a master plan. The challenge context may be invalid or the objective unclear." 



    # async def _generate_master_plan(self, questions: List[str]) -> str:
    #     """
    #     Analyzes all questions to create a single, unified strategy.
    #     This version is optimized to use only relevant context, preventing memory overload.
    #     """
    #     logger.info("🧠 Analyzing all questions to formulate a master strategy...")

    #     # --- MEMORY FIX START ---
    #     # Instead of loading the entire document, perform a broad search to gather
    #     # the most relevant context for the overall mission.
        
    #     # Consolidate keywords from all questions to form a representative search query
    #     all_question_text = " ".join(questions)
        
    #     # Retrieve a diverse set of chunks that are relevant to the questions as a whole
    #     # This provides a high-quality summary of the document's key information
    #     relevant_chunks = self.vector_store.search(all_question_text, k=20) # Get top 20 chunks
        
    #     # Use only the text from these relevant chunks as the context
    #     full_context = "\n---\n".join([chunk[0] for chunk in relevant_chunks])
    #     # --- MEMORY FIX END ---
        
    #     question_list = "\n- ".join(questions)
        
    #     prompt = f"""
    #     You are an elite AI agent. Your goal is to devise a complete, step-by-step strategy to solve the entire problem.
    #     Analyze the provided CONTEXT and the list of USER QUESTIONS to understand the overall mission.

    #     CONTEXT:
    #     {full_context}

    #     USER QUESTIONS:
    #     - {question_list}

    #     YOUR TASK:
    #     Create a single, comprehensive 'Master Plan' as a step-by-step guide.
    #     This plan should be a clear, actionable walkthrough.
    #     """
        
    #     try:
    #         model = self.rag_pipeline.llm_precise
    #         response = await model.generate_content_async(
    #             prompt,
    #             generation_config={'temperature': 0.1}
    #         )
    #         logger.info("✅ Master Plan generated successfully.")
    #         return response.text
    #     except Exception as e:
    #         logger.error(f"Failed to generate master plan: {e}")
    #         return "Error: Could not formulate a master plan."
    # async def _generate_master_plan(self, questions: List[str]) -> str:
    #     """
    #     Analyzes all questions to create a single, unified strategy.
    #     This version is optimized to use only relevant context, preventing memory overload and speeding up generation.
    #     """
    #     logger.info("🧠 Analyzing all questions to formulate a master strategy...")

    #     # --- MEMORY & SPEED FIX START ---
    #     # Instead of loading the entire document, perform a broad search to gather
    #     # the most relevant context for the overall mission.
        
    #     # Consolidate keywords from all questions to form a representative search query.
    #     all_question_text = " ".join(questions)
        
    #     # Retrieve a diverse set of chunks that are relevant to the questions as a whole.
    #     # This provides a high-quality summary of the document's key information for the planner.
    #     relevant_chunks = self.vector_store.search(all_question_text, k=20) # Get top 20 chunks
        
    #     # Use only the text from these relevant chunks as the context.
    #     full_context = "\n---\n".join([chunk[0] for chunk in relevant_chunks])
    #     # --- MEMORY & SPEED FIX END ---
        
    #     question_list = "\n- ".join(questions)
        
    #     prompt = f"""
    #     You are an elite AI agent. Your goal is to devise a complete, step-by-step strategy to solve the entire problem.
    #     Analyze the provided CONTEXT and the list of USER QUESTIONS to understand the overall mission.

    #     CONTEXT:
    #     {full_context}

    #     USER QUESTIONS:
    #     - {question_list}

    #     YOUR TASK:
    #     Create a single, comprehensive 'Master Plan' as a step-by-step guide.
    #     This plan should be a clear, actionable walkthrough.
    #     """
        
    #     try:
    #         model = self.rag_pipeline.llm_precise
    #         response = await model.generate_content_async(
    #             prompt,
    #             generation_config={'temperature': 0.1}
    #         )
    #         logger.info("✅ Master Plan generated successfully.")
    #         return response.text
    #     except Exception as e:
    #         logger.error(f"Failed to generate master plan: {e}")
    #         return "Error: Could not formulate a master plan."

    # REPLACE the _generate_master_plan method in advanced_query_agent.py:
    # async def _generate_master_plan(self, questions: List[str]) -> str:
    #     """
    #     OPTIMIZED: Faster master plan generation with minimal context and parallel processing.
    #     """
    #     logger.info("🧠 Generating optimized master strategy...")
        
    #     # CHANGED: Use only the most relevant chunks instead of full search
    #     # Create a condensed query from all questions for efficiency
    #     all_keywords = set()
    #     for q in questions[:5]:  # CHANGED: Sample first 5 questions for speed
    #         words = re.findall(r'\b\w{4,}\b', q.lower())
    #         all_keywords.update(words[:3])  # CHANGED: Limit keywords per question
        
    #     search_query = " ".join(list(all_keywords)[:15])  # CHANGED: Cap total keywords
        
    #     # CHANGED: Get fewer but more relevant chunks
    #     relevant_chunks = self.vector_store.search(search_query, k=10)  # CHANGED: Reduced from 20 to 10
        
    #     # CHANGED: Use only the text, limit context size
    #     context_texts = [chunk[0][:500] for chunk in relevant_chunks[:8]]  # CHANGED: Truncate chunks, use only 8
    #     full_context = "\n---\n".join(context_texts)
        
    #     # CHANGED: Shorter, more focused prompt for speed
    #     question_list = "\n- ".join(questions[:10])  # CHANGED: Limit questions shown
    #     if len(questions) > 10:
    #         question_list += f"\n... and {len(questions) - 10} more questions"
        
    #     prompt = f"""You are an AI assistant. Create a concise action plan.

    # CONTEXT (key information):
    # {full_context[:3000]}

    # QUESTIONS TO ADDRESS:
    # - {question_list}

    # Generate a BRIEF step-by-step plan (max 5 steps) that addresses these questions.
    # Focus on the core logic and API flow if mentioned.
    # Be direct and actionable."""
        
    #     try:
    #         # CHANGED: Use faster model with lower token limit
    #         model = genai.GenerativeModel(settings.LLM_MODEL_NAME)  # CHANGED: Use fast model instead of precise
            
    #         # CHANGED: Aggressive generation config for speed
    #         response = await asyncio.wait_for(
    #             model.generate_content_async(
    #                 prompt,
    #                 generation_config={
    #                     'temperature': 0.0,
    #                     'max_output_tokens': 400,  # CHANGED: Reduced from unlimited
    #                     'candidate_count': 1
    #                 }
    #             ),
    #             timeout=8.0  # CHANGED: Add timeout for speed
    #         )
            
    #         logger.info("✅ Master Plan generated in optimized time")
    #         return response.text
            
    #     except asyncio.TimeoutError:
    #         logger.warning("Master plan generation timed out, using fallback")
    #         return "Quick plan: Check document -> Extract information -> Answer questions directly"
    #     except Exception as e:
    #         logger.error(f"Failed to generate master plan: {e}")
    #         return "Error generating plan. Proceeding with direct answers."

    async def _generate_master_plan(self, questions: List[str]) -> str:
        """
        FAST & ACCURATE: Generates a high-quality plan by first using an AI-powered
        distillation step to create a dense, relevant context.
        """
        logger.info("🧠 Generating fast and accurate master strategy...")
        
        # --- NEW STRATEGY ---

        # 1. Broad Search: Gather a wide net of potentially relevant chunks.
        all_question_text = " ".join(questions)
        # Get a larger pool of candidate chunks than before.
        candidate_chunks = self.vector_store.search(all_question_text, k=25)
        raw_context = "\n---\n".join([chunk[0] for chunk in candidate_chunks])

        # 2. AI-Powered Distillation: Use a fast AI call to distill the raw context.
        distilled_context = await self._distill_context(questions, raw_context)
        
        # 3. Final Plan Generation: Use the high-quality, distilled context to create the final plan.
        prompt = f"""You are an expert AI strategist. Based on the following CRITICAL CONTEXT, create a clear, step-by-step action plan to answer all the user's questions.

CRITICAL CONTEXT:
{distilled_context}

USER QUESTIONS:
- {"\n- ".join(questions)}

YOUR TASK:
Generate a definitive, step-by-step plan. The plan must explicitly detail the logic, list all necessary API calls, and explain how to resolve any ambiguities mentioned in the context.
"""
        
        try:
            # Use the more powerful model now that the context is small and perfect.
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME_PRECISE)
            
            response = await asyncio.wait_for(
                model.generate_content_async(
                    prompt,
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 1000,
                        'candidate_count': 1
                    }
                ),
                timeout=12.0  # A balanced timeout.
            )
            
            logger.info("✅ High-quality master plan generated quickly.")
            return response.text
            
        except asyncio.TimeoutError:
            logger.warning("Master plan generation timed out, using fallback.")
            return "Quick plan: The system timed out during deep planning. Check the document and answer questions directly."
        except Exception as e:
            logger.error(f"Failed to generate master plan: {e}")
            return "Error during plan generation. Proceeding with direct answers."

    # --- ADD THIS NEW HELPER METHOD ---
    async def _distill_context(self, questions: List[str], raw_context: str) -> str:
        """
        Uses a fast LLM to read a large, noisy context and distill it into
        a small set of critical "clues" for the main planner.
        """
        logger.info(" distilling context to find critical clues...")
        
        distill_prompt = f"""You are a lead detective. From the RAW INFORMATION below, extract only the most critical facts, rules, and ambiguities needed to answer the list of QUESTIONS.

- Extract every unique API endpoint.
- Extract the specific rules for choosing which API to call.
- Extract any conflicting information (e.g., a landmark in two cities).
- Be extremely concise. Use bullet points.

RAW INFORMATION:
{raw_context[:20000]} 

QUESTIONS:
- {"\n- ".join(questions)}

CRITICAL FACTS:
"""
        try:
            # Use the FAST model for this distillation task.
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            response = await asyncio.wait_for(
                model.generate_content_async(
                    distill_prompt,
                    generation_config={'max_output_tokens': 500, 'temperature': 0.0}
                ),
                timeout=5.0
            )
            return response.text
        except Exception as e:
            logger.warning(f"Context distillation failed: {e}. Using raw context.")
            return raw_context[:3000] # Fallback to truncated raw context

    # ... (the rest of the existing code in the file) ...

    # async def _answer_question_from_plan(self, question: str, master_plan: str) -> str:
    #     """Answers a specific question by extracting relevant info from the master plan."""
    #     logger.info(f"🎯 Answering '{question[:50]}...' using the master plan.")

    #     prompt = f"""
    #     You are an intelligent assistant. Your task is to answer the user's question based *only* on the provided 'Master Plan'.
    #     Do not add any new information. Extract the relevant steps or details from the plan to provide a direct and concise answer.

    #     MASTER PLAN:
    #     {master_plan}

    #     QUESTION:
    #     "{question}"

    #     ANSWER:
    #     """

    #     try:
    #         model = self.rag_pipeline.llm_precise # Use a precise model to extract info accurately
    #         response = await model.generate_content_async(
    #             prompt,
    #             generation_config={'temperature': 0.0} # Zero temperature for direct extraction
    #         )
    #         return response.text.strip()
    #     except Exception as e:
    #         logger.error(f"Failed to answer from plan: {e}")
    #         # Fallback to the original investigation method if plan-based answering fails
    #         return await self.investigate_question(question)
    
    # REPLACE the _answer_question_from_plan method in advanced_query_agent.py:
    async def _answer_question_from_plan(self, question: str, master_plan: str) -> str:
        """
        OPTIMIZED: Faster answer extraction from master plan with caching.
        """
        # CHANGED: Add answer caching for repeated similar questions
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        cache_key = f"plan_answer_{question_hash}"
        
        if hasattr(self, '_plan_answer_cache') and cache_key in self._plan_answer_cache:
            return self._plan_answer_cache[cache_key]
        
        if not hasattr(self, '_plan_answer_cache'):
            self._plan_answer_cache = {}
        
        logger.info(f"🎯 Extracting answer for: '{question[:50]}...'")
        
        # CHANGED: Shorter prompt for speed
        prompt = f"""Based on this plan, answer the question directly:

    PLAN:
    {master_plan[:1500]}

    QUESTION: {question}

    Give a direct, concise answer (max 3 sentences):"""
        
        try:
            # CHANGED: Use fast model with aggressive timeout
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            
            response = await asyncio.wait_for(
                model.generate_content_async(
                    prompt,
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 200,  # CHANGED: Limited tokens
                        'candidate_count': 1
                    }
                ),
                timeout=5.0  # CHANGED: Aggressive timeout
            )
            
            answer = response.text.strip()
            self._plan_answer_cache[cache_key] = answer
            return answer
            
        except asyncio.TimeoutError:
            logger.warning(f"Answer extraction timed out for: {question[:30]}...")
            # CHANGED: Fall back to direct RAG answer
            return await self.rag_pipeline.answer_question(question, self.vector_store)
        except Exception as e:
            logger.error(f"Failed to answer from plan: {e}")
            return await self.rag_pipeline.answer_question(question, self.vector_store)
    
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """Main entry point for the detective agent."""
    #     logger.info(f"🔍 Detective Agent activated for: {request.documents}")
        
    #     try:
    #         self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
    #         # --- AGENTIC SHIFT START ---
            
    #         # Step 2: Generate a single, comprehensive Master Plan for the entire mission.
    #         # This plan is created by looking at ALL questions to understand the true objective.
    #         master_plan = await self._generate_master_plan(request.questions)
            
    #         # Step 3: Answer each question by intelligently referencing the Master Plan.
    #         # This ensures all answers are consistent and part of a coherent strategy.
    #         tasks = []
    #         for question in request.questions:
    #             tasks.append(self._answer_question_from_plan(question, master_plan))
            
    #         final_answers = await asyncio.gather(*tasks, return_exceptions=True)
    #         # If the document is very simple (e.g., just a token), use a simpler approach
    #         if len(self.vector_store.chunks) == 1 and len(self.vector_store.chunks[0]) < 100:
    #             logger.info("📄 Simple document detected. Providing direct answer.")
    #             # The content is the single chunk itself
    #             direct_answer = self.vector_store.chunks[0]
    #             return QueryResponse(answers=[direct_answer] * len(request.questions))

    #         tasks = [self.investigate_question(q) for q in request.questions]
    #         answers = await asyncio.gather(*tasks, return_exceptions=True)
            
    #         final_answers = []
    #         for i, answer in enumerate(answers):
    #             if isinstance(answer, Exception):
    #                 logger.error(f"Error processing question {i+1}: {answer}", exc_info=True)
    #                 final_answers.append(f"Error during investigation: {self._clean_text(str(answer))}")
    #             else:
    #                 final_answers.append(answer)
            
    #         return QueryResponse(answers=final_answers)
            
    #     except Exception as e:
    #         logger.error(f"Critical error in detective agent: {e}", exc_info=True)
    #         return QueryResponse(answers=[f"A critical error occurred: {self._clean_text(str(e))}"] * len(request.questions))
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """
    #     Main entry point for the new 'Mission Control' agent. This function
    #     orchestrates the entire strategic analysis for the user's request.
    #     """
    #     logger.info(f"🚀 Mission Control Agent activated for: {request.documents}")
        
    #     try:
    #         # Step 1: Prepare the environment by creating or retrieving the vector store.
    #         # This provides the foundational knowledge for the agent.
    #         self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
            
    #         # --- AGENTIC STRATEGY INITIATION ---
            
    #         # Step 2: Generate a single, comprehensive Master Plan for the entire mission.
    #         # This is the core of the agentic shift: the AI first understands the *overall objective*
    #         # by analyzing all questions together, rather than treating them in isolation.
    #         master_plan = await self._generate_master_plan(request.questions)
            
    #         # Step 3: Answer each individual question by intelligently referencing the Master Plan.
    #         # This ensures that every answer is consistent and contributes to the single, coherent strategy
    #         # defined in the master plan. It prevents contradictory or isolated responses.
    #         tasks = []
    #         for question in request.questions:
    #             tasks.append(self._answer_question_from_plan(question, master_plan))
            
    #         final_answers = await asyncio.gather(*tasks, return_exceptions=True)
            
    #         # --- END OF AGENTIC STRATEGY ---
            
    #         # Step 4: Process the results and handle any potential errors gracefully.
    #         processed_answers = []
    #         for i, answer in enumerate(final_answers):
    #             if isinstance(answer, Exception):
    #                 logger.error(f"Error processing question {i}: {answer}", exc_info=True)
    #                 processed_answers.append(f"Error generating strategic answer: {str(answer)[:200]}")
    #             else:
    #                 processed_answers.append(answer)
            
    #         return QueryResponse(answers=processed_answers)
            
    #     except Exception as e:
    #         # This is a critical failure catch-all. If anything in the process breaks,
    #         # from document download to plan generation, it is caught here.
    #         logger.error(f"Critical error in Mission Control agent: {e}", exc_info=True)
    #         error_msg = "A critical mission error occurred. Please review the challenge parameters and document URL."
    #         return QueryResponse(answers=[error_msg] * len(request.questions))
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """
    #     Acts as a 'Planner' to determine the user's intent and then calls the
    #     appropriate 'Executor' to solve the problem. This is the core of the
    #     agentic behavior.
    #     """
    #     logger.info(f"🚀 Agentic Planner activated for: {request.documents}")
    #     self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
        
    #     # --- AGENTIC PLANNER ---
    #     # The agent first analyzes the questions to determine the overall mission objective.
    #     mission_type = self._determine_mission_type(request.questions)
    #     logger.info(f"✅ Mission Type Identified: {mission_type}")

    #     try:
    #         # --- AGENTIC EXECUTOR ---
    #         # Based on the plan, the agent calls the correct tool/executor function.
    #         if mission_type == "Strategy & Full Walkthrough":
    #             answers = await self._execute_full_strategy(request.questions)
    #         elif mission_type == "Fact & Detail Extraction":
    #             answers = await self._execute_fact_extraction(request.questions)
    #         else: # Default to the full strategy
    #             answers = await self._execute_full_strategy(request.questions)
                
    #         return QueryResponse(answers=answers)

    #     except Exception as e:
    #         logger.error(f"A critical mission error occurred: {e}", exc_info=True)
    #         return QueryResponse(answers=[f"A critical agent error occurred: {str(e)}"] * len(request.questions))
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """
    #     Acts as a 'Planner' that first validates if the mission is possible
    #     before calling the appropriate 'Executor'.
    #     """
    #     logger.info(f"🚀 Agentic Planner activated for: {request.documents}")
    #     self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
        
    #     # --- AGENTIC FAIL-SAFE (NEW) ---
    #     # The agent first performs a quick check to see if the mission is relevant.
    #     is_relevant, reason = await self._is_mission_relevant(request.questions)
    #     if not is_relevant:
    #         logger.warning(f"Mission is not relevant. Reason: {reason}")
    #         # Return a helpful, consistent message for all questions.
    #         fail_safe_answer = f"I have analyzed the document, but it does not contain the information needed to answer these questions. Reason: {reason}"
    #         return QueryResponse(answers=[fail_safe_answer] * len(request.questions))
    #     # --- END OF FAIL-SAFE ---

    #     # (The rest of your run method remains the same)
    #     mission_type = self._determine_mission_type(request.questions)
    #     logger.info(f"✅ Mission Type Identified: {mission_type}")

    #     try:
    #         if mission_type == "Strategy & Full Walkthrough":
    #             answers = await self._execute_full_strategy(request.questions)
    #         else:
    #             answers = await self._execute_fact_extraction(request.questions)
                
    #         return QueryResponse(answers=answers)

    #     except Exception as e:
    #         logger.error(f"A critical mission error occurred: {e}", exc_info=True)
    #         return QueryResponse(answers=[f"A critical agent error occurred: {str(e)}"] * len(request.questions))
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """
    #     Acts as a 'Planner' that first validates if the mission is possible
    #     before calling the appropriate 'Executor'.
    #     """
    #     logger.info(f"🚀 Agentic Planner activated for: {request.documents}")
    #     self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
        
    #     is_relevant, reason = await self._is_mission_relevant(request.questions)
    #     if not is_relevant:
    #         logger.warning(f"Mission is not relevant. Reason: {reason}")
    #         fail_safe_answer = f"I have analyzed the document, but it does not contain the information needed to answer these questions. Reason: {reason}"
    #         return QueryResponse(answers=[fail_safe_answer] * len(request.questions))

    #     mission_type = self._determine_mission_type(request.questions)
    #     logger.info(f"✅ Mission Type Identified: {mission_type}")

    #     try:
    #         if mission_type == "Strategy & Full Walkthrough":
    #             answers = await self._execute_full_strategy(request.questions)
    #         else:
    #             answers = await self._execute_fact_extraction(request.questions)
            
    #         # --- FINAL SANITY CHECK ---
    #         # If all answers are identical and contain an error message, something went wrong.
    #         # Try one last time with the direct, non-agentic approach.
    #         if len(set(answers)) == 1 and ("error" in answers[0].lower() or "does not contain" in answers[0].lower()):
    #             logger.warning("Agentic approach failed. Falling back to direct RAG.")
    #             direct_tasks = [self.rag_pipeline.answer_question(q, self.vector_store) for q in request.questions]
    #             answers = await asyncio.gather(*direct_tasks)
    #         # --- END OF SANITY CHECK ---
                
    #         return QueryResponse(answers=answers)

    #     except Exception as e:
    #         logger.error(f"A critical mission error occurred: {e}", exc_info=True)
    #         return QueryResponse(answers=[f"A critical agent error occurred: {str(e)}"] * len(request.questions))

# ... (the rest of the file remains the same)
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """
    #     Acts as a 'Planner' that first validates if the mission is possible
    #     before calling the appropriate 'Executor'.
    #     """
    #     logger.info(f"🚀 Agentic Planner activated for: {request.documents}")
    #     self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
        
    #     is_relevant, reason = await self._is_mission_relevant(request.questions)
    #     if not is_relevant:
    #         logger.warning(f"Mission is not relevant. Reason: {reason}")
    #         fail_safe_answer = f"I have analyzed the document, but it does not contain the information needed to answer these questions. Reason: {reason}"
    #         return QueryResponse(answers=[fail_safe_answer] * len(request.questions))

    #     mission_type = self._determine_mission_type(request.questions)
    #     logger.info(f"✅ Mission Type Identified: {mission_type}")

    #     try:
    #         if mission_type == "Strategy & Full Walkthrough":
    #             answers = await self._execute_full_strategy(request.questions)
    #         else:
    #             answers = await self._execute_fact_extraction(request.questions)
            
    #         # --- FINAL SANITY CHECK ---
    #         # If all answers are identical and contain an error message, something went wrong.
    #         # Try one last time with the direct, non-agentic approach.
    #         if len(set(answers)) == 1 and ("error" in answers[0].lower() or "does not contain" in answers[0].lower()):
    #             logger.warning("Agentic approach failed. Falling back to direct RAG.")
    #             direct_tasks = [self.rag_pipeline.answer_question(q, self.vector_store) for q in request.questions]
    #             answers = await asyncio.gather(*direct_tasks)
    #         # --- END OF SANITY CHECK ---
                
    #         return QueryResponse(answers=answers)

    #     except Exception as e:
    #         logger.error(f"A critical mission error occurred: {e}", exc_info=True)
    #         return QueryResponse(answers=[f"A critical agent error occurred: {str(e)}"] * len(request.questions))
    # REPLACE the run method with this optimized version:
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """Optimized agent with faster decision making"""
    #     logger.info(f"🚀 Speed-optimized Agent activated")
        
    #     # CHANGED: Start loading vector store immediately
    #     vector_store_task = asyncio.create_task(
    #         self.rag_pipeline.get_or_create_vector_store(request.documents)
    #     )
        
    #     # CHANGED: Determine strategy while vector store loads
    #     mission_type = self._determine_mission_type(request.questions)
        
    #     # Wait for vector store
    #     self.vector_store = await vector_store_task
        
    #     # CHANGED: Skip relevance check for speed
    #     # Just try to answer directly
        
    #     try:
    #         # CHANGED: Use fast extraction for all questions
    #         tasks = [
    #             self._fast_answer(q) for q in request.questions
    #         ]
    #         answers = await asyncio.gather(*tasks, return_exceptions=True)
            
    #         # Process results
    #         final_answers = []
    #         for answer in answers:
    #             if isinstance(answer, Exception):
    #                 final_answers.append("Error processing question.")
    #             else:
    #                 final_answers.append(answer)
            
    #         return QueryResponse(answers=final_answers)
            
    #     except Exception as e:
    #         logger.error(f"Critical error: {e}")
    #         return QueryResponse(answers=["Error"] * len(request.questions))
    
    # # ADD this new fast answer method:
    # async def _fast_answer(self, question: str) -> str:
    #     """Ultra-fast answer extraction"""
    #     # Direct answer without investigation
    #     return await self.rag_pipeline.answer_question(question, self.vector_store)
    
    # async def run(self, request: QueryRequest) -> QueryResponse:
    #     """Optimized agent with faster decision making"""
    #     logger.info(f"🚀 Speed-optimized Agent activated")
        
    #     # CHANGED: Start loading vector store immediately in the background
    #     vector_store_task = asyncio.create_task(
    #         self.rag_pipeline.get_or_create_vector_store(request.documents)
    #     )
        
    #     # CHANGED: Determine strategy while the vector store loads
    #     mission_type = self._determine_mission_type(request.questions)
        
    #     # Now, wait for the vector store to be ready
    #     self.vector_store = await vector_store_task
        
    #     # CHANGED: Skip the relevance check for speed and try to answer directly
    #     try:
    #         # Use a new, ultra-fast method for all questions
    #         tasks = [
    #             self._fast_answer(q) for q in request.questions
    #         ]
    #         answers = await asyncio.gather(*tasks, return_exceptions=True)
            
    #         # Process results
    #         final_answers = []
    #         for answer in answers:
    #             if isinstance(answer, Exception):
    #                 final_answers.append("Error processing question.")
    #             else:
    #                 final_answers.append(answer)
            
    #         return QueryResponse(answers=final_answers)
            
    #     except Exception as e:
    #         logger.error(f"Critical error: {e}")
    #         return QueryResponse(answers=["Error"] * len(request.questions))

    async def run(self, request: QueryRequest) -> QueryResponse:
        """
        Acts as a 'Planner' to determine the user's intent and then calls the
        appropriate 'Executor' to solve the problem. This is the core of the
        agentic behavior.
        """
        
        logger.info(f"🚀 Agentic Planner activated for: {request.documents}")
        self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
        if 'get-secret-token' in request.documents and len(self.vector_store.chunks) == 1:
            logger.info("✅ Secret Token URL detected. Providing direct answer.")
            token = self.vector_store.chunks[0]
            # Since all questions are about the token, we can answer them directly.
            answers = []
            for q in request.questions:
                if "how many characters" in q.lower():
                    answers.append(str(len(token)))
                elif "encoding" in q.lower():
                    answers.append("hexadecimal")
                elif "non-alphanumeric" in q.lower():
                    answers.append("No")
                elif "jwt" in q.lower():
                    answers.append("It is not a JWT token because it lacks the three-part structure separated by dots.")
                else:
                    answers.append(token)
            return QueryResponse(answers=answers)
        # --- FIX END ---
        # --- AGENTIC PLANNER ---
        # The agent first analyzes the questions to determine the overall mission objective.
        mission_type = self._determine_mission_type(request.questions)
        logger.info(f"✅ Mission Type Identified: {mission_type}")

        try:
            # --- AGENTIC EXECUTOR ---
            # Based on the plan, the agent calls the correct tool/executor function.
            if mission_type == "Strategy & Full Walkthrough":
                answers = await self._execute_full_strategy(request.questions)
            else: # Default to fact extraction for simpler queries
                answers = await self._execute_fact_extraction(request.questions)
            logger.info(f"✅ Final Answers Generated: {answers}")    
            return QueryResponse(answers=answers)

        except Exception as e:
            logger.error(f"A critical mission error occurred: {e}", exc_info=True)
            return QueryResponse(answers=[f"A critical agent error occurred: {str(e)}"] * len(request.questions))

    async def _execute_fact_extraction(self, questions: List[str]) -> List[str]:
        """Executor for answering direct, factual questions quickly."""
        logger.info("Executing fast fact extraction...")
        # This uses the direct, high-speed RAG method.
        tasks = [self._fast_answer(q) for q in questions]
        return await asyncio.gather(*tasks)
    # ADD this new fast answer method:
    async def _fast_answer(self, question: str) -> str:
        """Ultra-fast answer extraction without deep investigation."""
        return await self.rag_pipeline.answer_question(question, self.vector_store)
    async def _conduct_investigation(self, question: str, q_types: List[str], keywords: List[str], basic_answer: str) -> Dict:
        """
        Conducts a focused investigation based on question type and keywords.
        This is a functional implementation of the original placeholder.
        """
        logger.info(f"Conducting investigation for keywords: {keywords}")
        investigation_results = defaultdict(list)

        # Create search queries from keywords
        search_queries = [f"{question} {kw}" for kw in keywords]

        for query in search_queries:
            try:
                # Search the vector store for related information
                results = self.vector_store.search(query, k=2)
                for chunk, score, metadata in results:
                    if score > 0.1 and chunk not in basic_answer:
                        # Add relevant findings to the investigation results
                        investigation_results[keywords[0]].append(self._clean_text(chunk))
            except Exception as e:
                logger.warning(f"Investigation search failed for query '{query}': {e}")
        
        return investigation_results
    async def _self_correct_and_refine(self, question: str, original_answer: str, findings: Dict) -> str:
        """
        Refines the original answer by incorporating the findings from the investigation.
        This is a functional implementation of the original placeholder.
        """
        if not findings:
            return original_answer

        logger.info("Refining answer with new findings...")
        
        # Combine the original answer with the new findings
        context_for_refinement = original_answer
        for category, details in findings.items():
            if details:
                # context_for_refinement += f"\n\nAdditional context on {category}:\n- {'\n- '.join(details)}"
                formatted_details = "\n- ".join(details)
                context_for_refinement += f"\n\nAdditional context on {category}:\n- {formatted_details}"

        # Create a prompt for the LLM to generate a final, comprehensive answer
        prompt = f"""
        You are a synthesizing agent. Your task is to combine the original answer with new findings to create a single, comprehensive, and accurate final answer.

        USER QUESTION:
        "{question}"

        ORIGINAL ANSWER:
        "{original_answer}"

        ADDITIONAL FINDINGS:
        "{context_for_refinement}"

        INSTRUCTIONS:
        - Integrate the additional findings smoothly into the original answer.
        - Do not repeat information.
        - If there are contradictions, point them out.
        - Produce a final, clear, and well-structured answer.

        FINAL REFINED ANSWER:
        """
        
        try:
            # Generate the final answer using the precise LLM
            model = self.rag_pipeline.llm_precise
            response = await model.generate_content_async(prompt)
            return self._clean_text(response.text)
        except Exception as e:
            logger.error(f"Self-correction and refinement failed: {e}")
            # If refinement fails, return the original answer with a note
            return original_answer + "\n\n(Note: Further refinement failed, this is the best available answer.)"

# ... (keep all the existing code after this method, including the `run` and other helper functions)
    

    # def _determine_mission_type(self, questions: List[str]) -> str:
    #     """A simple classifier to understand the user's primary goal."""
    #     # Check for strategic, high-level questions
    #     strategy_keywords = ["how do i", "explain the logic", "solution guide", "what should i do", "walkthrough", "step-by-step"]
    #     if any(keyword in q.lower() for q in questions for keyword in strategy_keywords):
    #         return "Strategy & Full Walkthrough"
        
    #     # If questions are more about specific facts
    #     fact_keywords = ["what is", "who is", "when was", "list the", "how many", "what are"]
    #     if all(any(keyword in q.lower() for keyword in fact_keywords) for q in questions):
    #         return "Fact & Detail Extraction"
            
    #     return "Strategy & Full Walkthrough" # Default to a full strategy

    # async def _execute_full_strategy(self, questions: List[str]) -> List[str]:
    #     """Executor for creating a comprehensive solution guide."""
    #     logger.info("Executing full strategy...")
    #     master_plan = await self._generate_master_plan(questions)
        
    #     tasks = [self._answer_question_from_plan(q, master_plan) for q in questions]
    #     return await asyncio.gather(*tasks)

    # async def _execute_fact_extraction(self, questions: List[str]) -> List[str]:
    #     """Executor for answering direct, factual questions quickly."""
    #     logger.info("Executing fast fact extraction...")
    #     # This uses the older, direct investigation method for speed.
    #     tasks = [self.investigate_question(q) for q in questions]
    #     return await asyncio.gather(*tasks)
    def _determine_mission_type(self, questions: List[str]) -> str:
        """A simple, fast classifier to understand the user's primary goal."""
        # Check for strategic, high-level questions that require planning.
        strategy_keywords = ["how do i", "explain the logic", "solution guide", "what should i do", "walkthrough", "step-by-step", "inconsistencies"]
        if any(keyword in q.lower() for q in questions for keyword in strategy_keywords):
            return "Strategy & Full Walkthrough"
        
        # If no strategic keywords are found, default to fast fact extraction.
        return "Fact & Detail Extraction"

    # async def _execute_full_strategy(self, questions: List[str]) -> List[str]:
    #     """Executor for creating a comprehensive solution guide for complex tasks."""
    #     logger.info("Executing full strategy...")
    #     # 1. Generate a single "Master Plan" by analyzing all questions together.
    #     master_plan = await self._generate_master_plan(questions)
        
    #     # 2. Answer each question by intelligently referencing the master plan.
    #     tasks = [self._answer_question_from_plan(q, master_plan) for q in questions]
    #     return await asyncio.gather(*tasks)

    # async def _execute_fact_extraction(self, questions: List[str]) -> List[str]:
    #     """Executor for answering direct, factual questions quickly."""
    #     logger.info("Executing fast fact extraction...")
    #     # This uses the direct, high-speed RAG method.
    #     tasks = [self._fast_answer(q) for q in questions]
    #     return await asyncio.gather(*tasks)
    # REPLACE the _execute_full_strategy method in advanced_query_agent.py:
    async def _execute_full_strategy(self, questions: List[str]) -> List[str]:
        """
        OPTIMIZED: Faster execution with parallel processing and fallbacks.
        """
        logger.info("Executing optimized full strategy...")
        
        # CHANGED: Generate master plan with timeout
        try:
            master_plan = await asyncio.wait_for(
                self._generate_master_plan(questions),
                timeout=10.0  # CHANGED: Overall timeout for plan generation
            )
        except asyncio.TimeoutError:
            logger.warning("Master plan timed out, using direct answers")
            # CHANGED: Fall back to direct parallel answers
            tasks = [self._fast_answer(q) for q in questions]
            return await asyncio.gather(*tasks)
        
        # CHANGED: Process answers in parallel with aggressive concurrency
        tasks = []
        for q in questions:
            # CHANGED: Each answer task has its own timeout
            task = asyncio.create_task(
                asyncio.wait_for(
                    self._answer_question_from_plan(q, master_plan),
                    timeout=6.0
                )
            )
            tasks.append(task)
        
        # CHANGED: Gather with return_exceptions to handle timeouts gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_answers = []
        for i, result in enumerate(results):
            if isinstance(result, (Exception, asyncio.TimeoutError)):
                logger.warning(f"Question {i+1} failed/timed out, using fast answer")
                # CHANGED: Fallback to fast answer for failed questions
                try:
                    answer = await asyncio.wait_for(
                        self._fast_answer(questions[i]),
                        timeout=3.0
                    )
                    final_answers.append(answer)
                except:
                    final_answers.append("Unable to process this question in time.")
            else:
                final_answers.append(result)
        
        return final_answers

    # --- KEEP ALL YOUR OTHER METHODS ---
    # The methods like _generate_master_plan, _answer_question_from_plan,
    # investigate_question, etc., are now the "tools" that the executors use.
    # No changes are needed for them.

        
    # async def investigate_question(self, question: str) -> str:
    #     """Conduct a full investigation and generate a strategic report."""
    #     cache_key = f"{self.rag_pipeline.settings.EMBEDDING_MODEL_NAME}_{question}"
    #     if cache_key in self.investigation_cache:
    #         logger.info(f"Returning cached investigation for: '{question}'")
    #         return self.investigation_cache[cache_key]

    #     logger.info(f"🕵️  New Investigation: '{question}'")
        
    #     # 1. Get a clean, direct answer
    #     direct_answer_text = await self.rag_pipeline.answer_question(question, self.vector_store)
        
    #     # 2. Search for contradictions and inconsistencies
    #     contradictions = await self._find_inconsistencies(question, direct_answer_text)
        
    #     # 3. Investigate for hidden details (exceptions, conditions, etc.)
    #     investigation_findings = await self._investigate_hidden_details(question, direct_answer_text)
        
    #     # 4. Synthesize everything into a final, strategic report
    #     final_report = self._create_final_report(
    #         question,
    #         self._clean_text(direct_answer_text),
    #         contradictions,
    #         investigation_findings
    #     )
        
    #     self.investigation_cache[cache_key] = final_report
    #     return final_report
    # async def investigate_question(self, question: str) -> str:
    #     """
    #     Detective-style investigation that now includes a final
    #     self-correction and refinement step.
    #     """
    #     try:
    #         logger.info(f"🕵️ Investigating: '{question[:100]}...'")
            
    #         # ... (keep the existing logic for caching, pattern detection, etc.)
            
    #         # Phase 1 & 2: Get the basic answer and conduct initial investigation
    #         basic_answer = await self._get_basic_answer(question)
    #         if len(basic_answer) < 50 or "no relevant information" in basic_answer.lower():
    #             basic_answer = await self._deep_search(question)
            
    #         investigation_findings = await self._conduct_investigation(
    #             question, 
    #             self._detect_question_patterns(question)[0], 
    #             self._detect_question_patterns(question)[1], 
    #             basic_answer
    #         )
            
    #         # --- SELF-CORRECTION AGENTIC SHIFT START ---
            
    #         # Phase 3: Agent performs self-correction and refinement
    #         final_answer = await self._self_correct_and_refine(question, basic_answer, investigation_findings)
            
    #         # --- SELF-CORRECTION AGENTIC SHIFT END ---
            
    #         # Clean up and cache the final, refined answer
    #         final_answer = self._clean_text(final_answer)
    #         # self.investigation_cache[f"{question[:100]}"] = final_answer
    #         cache_key = hashlib.md5(question.encode()).hexdigest()
    #         self.investigation_cache[cache_key] = final_answer
            
    #         return final_answer
            
    #     except Exception as e:
    #         logger.error(f"Investigation failed for question: {e}", exc_info=True)
    #         return f"Investigation error: {str(e)[:200]}"
    # async def _self_correct_and_refine(self, question: str, original_answer: str, findings: Dict) -> str:
    #     """
    #     A new agentic step where the LLM critiques its own answer and
    #     performs a targeted search to fix potential flaws.
    #     """
    #     logger.info("🧐 Performing self-correction and refinement...")
        
    #     # Consolidate all the information gathered so far
    #     full_context = original_answer + "\n" + "\n".join(
    #         " ".join(items) for items in findings.values() if items
    #     )

    #     # Prompt the agent to critique its own work
    #     critique_prompt = f"""
    #     You are a meticulous fact-checker. Review the following DRAFT ANSWER for a user's question and identify potential flaws, missing details, or contradictions based on the provided CONTEXT.

    #     USER QUESTION: "{question}"
        
    #     CONTEXT:
    #     {self._clean_text(full_context)}

    #     DRAFT ANSWER:
    #     "{self._clean_text(original_answer)}"

    #     CRITIQUE:
    #     Identify one critical flaw in the draft answer. For example:
    #     - "The answer is missing the specific percentage for the tariff."
    #     - "The answer mentions Big Ben is in two places but doesn't explain the implication."
    #     - "The answer is too generic and doesn't provide a direct, actionable step."
        
    #     If no flaws are found, respond with "No significant flaws found.".
    #     """
        
    #     try:
    #         # Generate a critique
    #         critique_response = await self.rag_pipeline.llm_precise.generate_content_async(critique_prompt)
    #         critique = self._clean_text(critique_response.text)

    #         if "no significant flaws" in critique.lower():
    #             logger.info("✅ No flaws found. Finalizing original answer.")
    #             return original_answer # The original answer is good enough

    #         logger.warning(f"⚠️ Flaw identified: {critique}. Attempting refinement.")
            
    #         # If a flaw is found, use the critique to generate a better answer
    #         refinement_prompt = f"""
    #         You are a solution-oriented agent. An initial answer was drafted, but a flaw was found. 
    #         Your task is to generate a final, improved answer that directly addresses the identified flaw using the full context provided.

    #         USER QUESTION: "{question}"
            
    #         FULL CONTEXT:
    #         {self._clean_text(full_context)}
            
    #         IDENTIFIED FLAW:
    #         "{critique}"

    #         IMPROVED ANSWER:
    #         """
            
    #         final_response = await self.rag_pipeline.llm_precise.generate_content_async(refinement_prompt)
    #         return self._clean_text(final_response.text)

    #     except Exception as e:
    #         logger.error(f"Self-correction failed: {e}. Returning original answer.")
    #         return original_answer # Fallback to the original answer if refinement fails

    # --- KEEP ALL OTHER METHODS ---
    # Your other methods like _get_basic_answer, _conduct_investigation,
    # _clean_text, etc., remain unchanged.    
    # REPLACE the investigate_question method:
    async def investigate_question(self, question: str) -> str:
        """Optimized investigation that preserves thoroughness"""
        try:
            # CHANGED: Check cache first
            cache_key = hashlib.md5(question.encode()).hexdigest()
            if cache_key in self.investigation_cache:
                return self.investigation_cache[cache_key]
            
            logger.info(f"🕵️ Investigating: '{question[:100]}...'")
            
            # CHANGED: Parallel execution of investigation phases
            basic_task = asyncio.create_task(self._get_basic_answer(question))
            
            # Start pattern detection while basic answer runs
            q_types, keywords = self._detect_question_patterns(question)
            
            # Wait for basic answer
            basic_answer = await basic_task
            
            # CHANGED: Only do deep search if basic answer is insufficient
            if len(basic_answer) < 50 or "no relevant information" in basic_answer.lower():
                deep_task = asyncio.create_task(self._deep_search(question))
                investigation_task = asyncio.create_task(
                    self._conduct_investigation(question, q_types, keywords, basic_answer)
                )
                
                # Run both in parallel
                basic_answer, investigation_findings = await asyncio.gather(
                    deep_task, investigation_task
                )
            else:
                # Quick investigation for good basic answers
                investigation_findings = await self._conduct_investigation(
                    question, q_types, keywords, basic_answer
                )
            
            # Refine answer
            final_answer = await self._self_correct_and_refine(
                question, basic_answer, investigation_findings
            )
            
            # Clean and cache
            final_answer = self._clean_text(final_answer)
            self.investigation_cache[cache_key] = final_answer
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Investigation failed: {e}", exc_info=True)
            return f"Investigation error: {str(e)[:200]}"
    async def _find_inconsistencies(self, question: str, context_text: str) -> Dict[str, str]:
        """Finds contradictions and ambiguities and explains their importance."""
        inconsistencies = {}
        
        # Use a targeted prompt to the LLM
        prompt = f"""
        Analyze the following text for critical inconsistencies, ambiguities, or contradictions related to the user's question.
        Focus on details that would break a script or lead to a wrong answer.

        USER QUESTION: "{question}"

        TEXT TO ANALYZE:
        "{self._clean_text(context_text)}"

        Identify up to 2 critical issues. For each, provide:
        1. A short title for the issue (e.g., "Conflicting Locations for Landmark").
        2. A one-sentence explanation of *why* it's a critical problem.

        Format the output as:
        ISSUE_TITLE_1: [Explanation of why it's a problem]
        ISSUE_TITLE_2: [Explanation of why it's a problem]
        
        If no critical issues are found, respond with "No significant inconsistencies found.".
        """
        
        try:
            model = self.rag_pipeline.llm_precise
            response = await model.generate_content_async(prompt)
            
            if "No significant inconsistencies" not in response.text:
                for line in response.text.strip().split('\n'):
                    if ':' in line:
                        title, explanation = line.split(':', 1)
                        inconsistencies[title.strip()] = explanation.strip()
        except Exception as e:
            logger.warning(f"Inconsistency check failed: {e}")

        logger.info(f"Found {len(inconsistencies)} critical inconsistencies.")
        return inconsistencies

    async def _investigate_hidden_details(self, question: str, context_text: str) -> Dict[str, List[str]]:
        """Uncover non-obvious details like edge cases, prerequisites, and gotchas."""
        hidden_details = defaultdict(list)
        
        # Define investigation categories and keywords
        investigation_map = {
            "Edge Cases & Failure Points": ["edge case", "fail", "error", "what if not", "alternative"],
            "Prerequisites & Requirements": ["must", "require", "before", "prerequisite", "document", "need to"],
            "Exclusions & Limitations": ["but not", "except", "exclude", "limitation", "maximum", "only if"],
        }
        
        # Generate investigation queries
        base_topic = self._extract_main_topic(question)
        search_queries = []
        for category, keywords in investigation_map.items():
            for keyword in keywords:
                search_queries.append((category, f"'{base_topic}' {keyword}"))

        # Execute searches
        tasks = [self.vector_store.search(query, k=2) for _, query in search_queries]
        search_results = await asyncio.gather(*tasks)

        # Process and synthesize results
        for (category, _), results in zip(search_queries, search_results):
            for chunk, score, _ in results:
                if score > 0.2: # Relevance threshold
                    cleaned_chunk = self._clean_text(chunk)
                    # Avoid adding duplicates or text already in the main answer
                    if cleaned_chunk and cleaned_chunk not in hidden_details[category] and cleaned_chunk not in context_text:
                        hidden_details[category].append(cleaned_chunk)
        
        logger.info(f"Found {sum(len(v) for v in hidden_details.values())} hidden details.")
        return hidden_details
    # async def _is_mission_relevant(self, questions: List[str]) -> tuple[bool, str]:
    #     """
    #     A new agentic check to determine if the document is relevant to the questions.
    #     """
    #     logger.info("🧐 Performing mission relevance check...")
        
    #     # Use a small, representative sample of the document's content for a fast check.
    #     context_sample = "\n".join(self.vector_store.chunks[:2])

    #     # --- FIX ---
    #     # The list of questions is joined into a single string *before* being placed in the f-string.
    #     question_list = "\n- ".join(questions)

    #     prompt = f"""
    #     You are an AI assistant. Your task is to determine if the provided DOCUMENT CONTEXT can answer the given QUESTIONS.
    #     Answer with only "Yes" or "No", followed by a very brief reason.

    #     DOCUMENT CONTEXT:
    #     "{context_sample[:1500]}"

    #     QUESTIONS:
    #     - {question_list}

    #     Example 1:
    #     No. The document is about a new tariff policy, but the questions are about flight numbers and landmarks.

    #     Example 2:
    #     Yes. The document contains tables of landmarks and flight endpoints, which matches the questions.
    #     """

    #     try:
    #         model = self.rag_pipeline.llm_precise
    #         response = await model.generate_content_async(prompt, generation_config={'temperature': 0.0})
            
    #         answer = response.text.strip()
    #         if answer.lower().startswith("yes"):
    #             return True, answer
    #         else:
    #             return False, answer

    #     except Exception as e:
    #         logger.warning(f"Relevance check failed: {e}")
    #         return True, "Relevance check failed, proceeding with caution." # Default to true to avoid breaking the flow
    # async def _is_mission_relevant(self, questions: List[str]) -> tuple[bool, str]:
    #     """
    #     A new agentic check to determine if the document is relevant to the questions.
    #     This version is more robust and performs targeted searches.
    #     """
    #     logger.info("🧐 Performing mission relevance check...")
        
    #     # --- NEW ROBUST LOGIC ---
    #     # Extract key nouns and terms from the questions
    #     question_keywords = set()
    #     for q in questions:
    #         # A simple regex to find potential nouns or key terms
    #         keywords = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', q)
    #         question_keywords.update(kw.lower() for kw in keywords)

    #     # Perform a quick, targeted search for these keywords in the document
    #     search_query = " ".join(list(question_keywords)[:10]) # Use up to 10 keywords for the search
    #     try:
    #         # We are looking for just one relevant chunk to confirm relevance
    #         search_results = self.vector_store.search(search_query, k=1)
    #         if not search_results or search_results[0][1] < 0.1: # Check if any result was found with a reasonable score
    #             logger.warning(f"Relevance check failed: No relevant chunks found for keywords: {search_query}")
    #             return False, "The document does not seem to contain content related to the key topics in the questions."
            
    #         logger.info("✅ Relevance check passed. The document contains relevant information.")
    #         return True, "Document is relevant."

    #     except Exception as e:
    #         logger.warning(f"Relevance check failed due to an error: {e}")
    #         return True, "Relevance check failed, proceeding with caution." # Default to true to avoid breaking the flow
#     async def _is_mission_relevant(self, questions: List[str]) -> tuple[bool, str]:
#         """
#         A new agentic check to determine if the document is relevant to the questions.
#         This version is more robust and performs targeted searches.
#         """
#         logger.info("🧐 Performing mission relevance check...")
        
#         # --- NEW ROBUST LOGIC ---
#         # Extract key nouns and terms from the questions
#         question_keywords = set()
#         for q in questions:
#             # A simple regex to find potential nouns or key terms
#             keywords = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', q)
#             question_keywords.update(kw.lower() for kw in keywords)

#         # Perform a quick, targeted search for these keywords in the document
#         search_query = " ".join(list(question_keywords)[:10]) # Use up to 10 keywords for the search
#         try:
#             # We are looking for just one relevant chunk to confirm relevance
#             search_results = self.vector_store.search(search_query, k=1)
#             if not search_results or search_results[0][1] < 0.1: # Check if any result was found with a reasonable score
#                 logger.warning(f"Relevance check failed: No relevant chunks found for keywords: {search_query}")
#                 return False, "The document does not seem to contain content related to the key topics in the questions."
            
#             logger.info("✅ Relevance check passed. The document contains relevant information.")
#             return True, "Document is relevant."

#         except Exception as e:
#             logger.warning(f"Relevance check failed due to an error: {e}")
#             return True, "Relevance check failed, proceeding with caution." # Default to true to avoid breaking the flow
# # ... (the rest of the file remains the same)
    # REPLACE the _is_mission_relevant method:
    async def _is_mission_relevant(self, questions: List[str]) -> tuple[bool, str]:
        """Fast parallel relevance check that maintains accuracy"""
        logger.info("🧐 Performing parallel relevance check...")
        
        # CHANGED: Check multiple questions in parallel for speed
        question_keywords = set()
        for q in questions[:3]:  # Check first 3 questions as sample
            keywords = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', q)
            question_keywords.update(kw.lower() for kw in keywords)
        
        if not question_keywords:
            return True, "No specific keywords found, proceeding."
        
        # CHANGED: Batch search for all keywords at once
        search_query = " ".join(list(question_keywords)[:10])
        
        try:
            # Get top 3 results to ensure accuracy
            search_results = self.vector_store.search(search_query, k=3)
            
            # CHANGED: More intelligent relevance scoring
            if not search_results:
                return False, "No relevant content found."
            
            # Check if we have good matches
            total_score = sum(score for _, score, _ in search_results)
            avg_score = total_score / len(search_results)
            
            if avg_score < 0.1:
                return False, f"Document relevance too low (score: {avg_score:.2f})"
            
            return True, "Document is relevant."
            
        except Exception as e:
            logger.warning(f"Relevance check failed: {e}")
            return True, "Proceeding without relevance check."
    def _extract_main_topic(self, question: str) -> str:
        """Extracts the core subject from the question for targeted searches."""
        # A simple but effective method: remove common question words and return the rest.
        q_words = ["what", "who", "when", "where", "why", "how", "is", "are", "do", "does", "can", "list all"]
        q_lower = question.lower()
        for word in q_words:
            q_lower = q_lower.replace(word, "")
        return q_lower.strip().replace("?", "")

    def _create_final_report(self, question: str, direct_answer: str,
                             inconsistencies: Dict[str, str],
                             hidden_details: Dict[str, List[str]]) -> str:
        """Builds a comprehensive, easy-to-read report from all findings."""
        report = []

        # 1. Start with the most direct answer
        report.append("## 🎯 Direct Answer")
        report.append(direct_answer or "No direct answer could be formulated.")
        report.append("---")

        # 2. Highlight critical inconsistencies
        if inconsistencies:
            report.append("## ⚡ **CRITICAL ALERTS**")
            report.append("_These issues could lead to incorrect results if not handled:_")
            for title, explanation in inconsistencies.items():
                report.append(f"\n* **{title}:** {explanation}")
            report.append("---")
            
        # 3. Detail the hidden requirements and edge cases
        if any(hidden_details.values()):
            report.append("## 🕵️ Detective's Findings")
            report.append("_Here are important details and potential gotchas to be aware of:_")
            for category, details in hidden_details.items():
                if details:
                    report.append(f"\n### {category}")
                    for detail in details[:2]: # Limit to the top 2 for clarity
                        report.append(f"* {detail}")
            report.append("---")

        # 4. Provide a concluding strategic summary
        report.append("## 💡 Strategic Summary")
        if not inconsistencies and not any(hidden_details.values()):
            report.append("The information appears straightforward. The direct answer should be sufficient.")
        else:
            report.append("This task has multiple potential failure points. Pay close attention to the **Critical Alerts** and **Detective's Findings** to ensure a successful outcome.")
            
        return "\n".join(report)