# app/agents/advanced_query_agent.py
import logging
import asyncio
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import html

from app.models.query import QueryRequest, QueryResponse
from app.core.rag_pipeline import HybridRAGPipeline, OptimizedVectorStore

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
    async def _generate_master_plan(self, questions: List[str]) -> str:
        """Analyzes all questions to create a single, unified strategy."""
        logger.info("🧠 Analyzing all questions to formulate a master strategy...")

        # Consolidate all available text from the vector store to use as context
        full_context = "\n---\n".join(self.vector_store.chunks)
        
        # --- FIX ---
        # The list of questions is joined into a single string *before* being placed in the f-string.
        question_list = "\n- ".join(questions)
        
        # Create a prompt that asks the LLM to think like a hackathon winner
        prompt = f"""
        You are an elite AI agent in a high-stakes, interactive programming challenge.
        Your goal is to devise a complete, step-by-step strategy to solve the entire problem, not just answer individual questions.
        Analyze the provided context and the list of user questions to understand the overall mission.

        CONTEXT:
        {full_context}

        USER QUESTIONS (use these to understand the mission's scope):
        - {question_list}

        YOUR TASK:
        Create a single, comprehensive 'Master Plan' as a step-by-step guide to win the challenge.
        This plan should be a clear, actionable walkthrough. Identify critical steps, potential pitfalls, and the final objective.
        Be smart, anticipate the required sequence of actions, and explain the logic.
        """
        
        try:
            # Use the most powerful model for strategic planning
            model = self.rag_pipeline.llm_precise
            response = await model.generate_content_async(
                prompt,
                generation_config={'temperature': 0.1} # Low temperature for factual, deterministic plans
            )
            logger.info("✅ Master Plan generated successfully.")
            return response.text
        except Exception as e:
            logger.error(f"Failed to generate master plan: {e}")
            return "Error: Could not formulate a master plan. The challenge context may be invalid or the objective unclear."        
    async def _answer_question_from_plan(self, question: str, master_plan: str) -> str:
        """Answers a specific question by extracting relevant info from the master plan."""
        logger.info(f"🎯 Answering '{question[:50]}...' using the master plan.")

        prompt = f"""
        You are an intelligent assistant. Your task is to answer the user's question based *only* on the provided 'Master Plan'.
        Do not add any new information. Extract the relevant steps or details from the plan to provide a direct and concise answer.

        MASTER PLAN:
        {master_plan}

        QUESTION:
        "{question}"

        ANSWER:
        """

        try:
            model = self.rag_pipeline.llm_precise # Use a precise model to extract info accurately
            response = await model.generate_content_async(
                prompt,
                generation_config={'temperature': 0.0} # Zero temperature for direct extraction
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to answer from plan: {e}")
            # Fallback to the original investigation method if plan-based answering fails
            return await self.investigate_question(question)
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
    async def run(self, request: QueryRequest) -> QueryResponse:
        """
        Acts as a 'Planner' to determine the user's intent and then calls the
        appropriate 'Executor' to solve the problem. This is the core of the
        agentic behavior.
        """
        logger.info(f"🚀 Agentic Planner activated for: {request.documents}")
        self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
        
        # --- AGENTIC PLANNER ---
        # The agent first analyzes the questions to determine the overall mission objective.
        mission_type = self._determine_mission_type(request.questions)
        logger.info(f"✅ Mission Type Identified: {mission_type}")

        try:
            # --- AGENTIC EXECUTOR ---
            # Based on the plan, the agent calls the correct tool/executor function.
            if mission_type == "Strategy & Full Walkthrough":
                answers = await self._execute_full_strategy(request.questions)
            elif mission_type == "Fact & Detail Extraction":
                answers = await self._execute_fact_extraction(request.questions)
            else: # Default to the full strategy
                answers = await self._execute_full_strategy(request.questions)
                
            return QueryResponse(answers=answers)

        except Exception as e:
            logger.error(f"A critical mission error occurred: {e}", exc_info=True)
            return QueryResponse(answers=[f"A critical agent error occurred: {str(e)}"] * len(request.questions))

    def _determine_mission_type(self, questions: List[str]) -> str:
        """A simple classifier to understand the user's primary goal."""
        # Check for strategic, high-level questions
        strategy_keywords = ["how do i", "explain the logic", "solution guide", "what should i do", "walkthrough", "step-by-step"]
        if any(keyword in q.lower() for q in questions for keyword in strategy_keywords):
            return "Strategy & Full Walkthrough"
        
        # If questions are more about specific facts
        fact_keywords = ["what is", "who is", "when was", "list the", "how many", "what are"]
        if all(any(keyword in q.lower() for keyword in fact_keywords) for q in questions):
            return "Fact & Detail Extraction"
            
        return "Strategy & Full Walkthrough" # Default to a full strategy

    async def _execute_full_strategy(self, questions: List[str]) -> List[str]:
        """Executor for creating a comprehensive solution guide."""
        logger.info("Executing full strategy...")
        master_plan = await self._generate_master_plan(questions)
        
        tasks = [self._answer_question_from_plan(q, master_plan) for q in questions]
        return await asyncio.gather(*tasks)

    async def _execute_fact_extraction(self, questions: List[str]) -> List[str]:
        """Executor for answering direct, factual questions quickly."""
        logger.info("Executing fast fact extraction...")
        # This uses the older, direct investigation method for speed.
        tasks = [self.investigate_question(q) for q in questions]
        return await asyncio.gather(*tasks)

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
    async def investigate_question(self, question: str) -> str:
        """
        Detective-style investigation that now includes a final
        self-correction and refinement step.
        """
        try:
            logger.info(f"🕵️ Investigating: '{question[:100]}...'")
            
            # ... (keep the existing logic for caching, pattern detection, etc.)
            
            # Phase 1 & 2: Get the basic answer and conduct initial investigation
            basic_answer = await self._get_basic_answer(question)
            if len(basic_answer) < 50 or "no relevant information" in basic_answer.lower():
                basic_answer = await self._deep_search(question)
            
            investigation_findings = await self._conduct_investigation(
                question, 
                self._detect_question_patterns(question)[0], 
                self._detect_question_patterns(question)[1], 
                basic_answer
            )
            
            # --- SELF-CORRECTION AGENTIC SHIFT START ---
            
            # Phase 3: Agent performs self-correction and refinement
            final_answer = await self._self_correct_and_refine(question, basic_answer, investigation_findings)
            
            # --- SELF-CORRECTION AGENTIC SHIFT END ---
            
            # Clean up and cache the final, refined answer
            final_answer = self._clean_text(final_answer)
            self.investigation_cache[f"{question[:100]}"] = final_answer
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Investigation failed for question: {e}", exc_info=True)
            return f"Investigation error: {str(e)[:200]}"
    async def _self_correct_and_refine(self, question: str, original_answer: str, findings: Dict) -> str:
        """
        A new agentic step where the LLM critiques its own answer and
        performs a targeted search to fix potential flaws.
        """
        logger.info("🧐 Performing self-correction and refinement...")
        
        # Consolidate all the information gathered so far
        full_context = original_answer + "\n" + "\n".join(
            " ".join(items) for items in findings.values() if items
        )

        # Prompt the agent to critique its own work
        critique_prompt = f"""
        You are a meticulous fact-checker. Review the following DRAFT ANSWER for a user's question and identify potential flaws, missing details, or contradictions based on the provided CONTEXT.

        USER QUESTION: "{question}"
        
        CONTEXT:
        {self._clean_text(full_context)}

        DRAFT ANSWER:
        "{self._clean_text(original_answer)}"

        CRITIQUE:
        Identify one critical flaw in the draft answer. For example:
        - "The answer is missing the specific percentage for the tariff."
        - "The answer mentions Big Ben is in two places but doesn't explain the implication."
        - "The answer is too generic and doesn't provide a direct, actionable step."
        
        If no flaws are found, respond with "No significant flaws found.".
        """
        
        try:
            # Generate a critique
            critique_response = await self.rag_pipeline.llm_precise.generate_content_async(critique_prompt)
            critique = self._clean_text(critique_response.text)

            if "no significant flaws" in critique.lower():
                logger.info("✅ No flaws found. Finalizing original answer.")
                return original_answer # The original answer is good enough

            logger.warning(f"⚠️ Flaw identified: {critique}. Attempting refinement.")
            
            # If a flaw is found, use the critique to generate a better answer
            refinement_prompt = f"""
            You are a solution-oriented agent. An initial answer was drafted, but a flaw was found. 
            Your task is to generate a final, improved answer that directly addresses the identified flaw using the full context provided.

            USER QUESTION: "{question}"
            
            FULL CONTEXT:
            {self._clean_text(full_context)}
            
            IDENTIFIED FLAW:
            "{critique}"

            IMPROVED ANSWER:
            """
            
            final_response = await self.rag_pipeline.llm_precise.generate_content_async(refinement_prompt)
            return self._clean_text(final_response.text)

        except Exception as e:
            logger.error(f"Self-correction failed: {e}. Returning original answer.")
            return original_answer # Fallback to the original answer if refinement fails

    # --- KEEP ALL OTHER METHODS ---
    # Your other methods like _get_basic_answer, _conduct_investigation,
    # _clean_text, etc., remain unchanged.    

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