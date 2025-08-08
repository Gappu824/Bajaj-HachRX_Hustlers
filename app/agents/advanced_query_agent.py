# app/agents/advanced_query_agent.py
import logging
import asyncio
import re
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from app.models.query import QueryRequest, QueryResponse, DetailedAnswer, SourceClause
from app.core.rag_pipeline import HybridRAGPipeline, OptimizedVectorStore

logger = logging.getLogger(__name__)

class AdvancedQueryAgent:
    """
    A detective-style agent that investigates beyond explicit questions,
    finding exceptions, contradictions, and related insights.
    """

    def __init__(self, rag_pipeline: HybridRAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.vector_store: OptimizedVectorStore = None
        self.investigation_history = {}  # Track all findings
        self.investigation_graph = defaultdict(list)  # Related findings
        
        # Detective patterns for different question types
        self.detective_patterns = {
            'time_period': {
                'triggers': ['period', 'days', 'months', 'years', 'time', 'duration', 'deadline', 'when'],
                'investigate': ['exception', 'extension', 'grace period', 'holiday', 'business days', 
                               'calendar days', 'renewal', 'expiry', 'early', 'late', 'penalty']
            },
            'money': {
                'triggers': ['cost', 'price', 'fee', 'amount', 'payment', 'premium', 'charge', 
                           'salary', 'compensation', 'reimbursement', 'deductible'],
                'investigate': ['tax', 'penalty', 'interest', 'late fee', 'discount', 'maximum', 
                               'minimum', 'limit', 'co-pay', 'excluded', 'additional charges']
            },
            'coverage': {
                'triggers': ['covered', 'eligible', 'included', 'benefit', 'claim', 'insurance'],
                'investigate': ['excluded', 'exception', 'not covered', 'pre-existing', 'waiting period',
                               'maximum benefit', 'sub-limit', 'conditions', 'requirements', 'documentation']
            },
            'eligibility': {
                'triggers': ['eligible', 'qualify', 'requirement', 'criteria', 'who can'],
                'investigate': ['age limit', 'documentation', 'proof', 'minimum', 'maximum', 
                               'excluded', 'special cases', 'grandfather clause']
            },
            'process': {
                'triggers': ['how to', 'process', 'procedure', 'steps', 'apply', 'claim', 'submit'],
                'investigate': ['deadline', 'required documents', 'forms', 'approval', 'rejection',
                               'appeal', 'review', 'turnaround time', 'contact']
            },
            'comparison': {
                'triggers': ['compare', 'difference', 'versus', 'vs', 'better', 'choose'],
                'investigate': ['advantages', 'disadvantages', 'limitations', 'benefits', 
                               'costs', 'restrictions', 'special features']
            }
        }

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Main entry point for the detective agent."""
        logger.info(f"🔍 Detective Agent activated for document: {request.documents}")
        
        # Step 1: Ensure the document is processed
        self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
        
        # Step 2: Process questions with detective mindset
        tasks = [self.investigate_question(q) for q in request.questions]
        answers = await asyncio.gather(*tasks)
        
        return QueryResponse(answers=answers)

    async def investigate_question(self, question: str) -> str:
        """
        Detective-style investigation that goes beyond the explicit question.
        """
        logger.info(f"🕵️ Beginning investigation for: '{question}'")
        
        # Detect question type and patterns
        question_type, keywords = self._detect_question_pattern(question)
        logger.info(f"📊 Detected pattern: {question_type}, Keywords: {keywords}")
        
        # Phase 1: Get the basic answer
        basic_answer = await self._get_basic_answer(question)
        
        # Phase 2: Detective investigation based on pattern
        investigation_findings = await self._conduct_investigation(
            question, question_type, keywords, basic_answer
        )
        
        # Phase 3: Check for contradictions and inconsistencies
        contradictions = await self._find_contradictions(question, investigation_findings)
        
        # Phase 4: Find related edge cases
        edge_cases = await self._investigate_edge_cases(question, question_type)
        
        # Phase 5: Synthesize comprehensive detective report
        final_answer = await self._create_detective_report(
            question, basic_answer, investigation_findings, contradictions, edge_cases
        )
        
        return final_answer

    def _detect_question_pattern(self, question: str) -> Tuple[str, List[str]]:
        """Detect the type of question and extract key terms."""
        question_lower = question.lower()
        detected_patterns = []
        all_keywords = []
        
        for pattern_type, pattern_info in self.detective_patterns.items():
            for trigger in pattern_info['triggers']:
                if trigger in question_lower:
                    detected_patterns.append(pattern_type)
                    all_keywords.extend(pattern_info['investigate'])
                    break
        
        # Default to general if no pattern detected
        if not detected_patterns:
            detected_patterns = ['general']
            all_keywords = ['exception', 'condition', 'requirement', 'limit', 'special case']
        
        return detected_patterns[0], list(set(all_keywords))

    async def _get_basic_answer(self, question: str) -> str:
        """Get the straightforward answer to the question."""
        logger.info("📝 Getting basic answer...")
        return await self.rag_pipeline.answer_question(question, self.vector_store)

    async def _conduct_investigation(self, question: str, question_type: str, 
                                    keywords: List[str], basic_answer: str) -> Dict[str, Any]:
        """Conduct detailed investigation based on question type."""
        findings = {
            'exceptions': [],
            'conditions': [],
            'related_info': [],
            'important_notes': [],
            'prerequisites': [],
            'limitations': []
        }
        
        # Investigate each keyword area
        investigation_queries = []
        
        # Create targeted investigation queries
        base_topic = self._extract_main_topic(question)
        
        for keyword in keywords[:10]:  # Limit to top 10 to avoid overwhelming
            investigation_queries.append(f"{base_topic} {keyword}")
        
        # Add specific investigation queries based on type
        if question_type == 'time_period':
            investigation_queries.extend([
                f"What happens if {base_topic} is delayed?",
                f"Are there extensions for {base_topic}?",
                f"What are the penalties related to {base_topic}?"
            ])
        elif question_type == 'money':
            investigation_queries.extend([
                f"What are the additional charges for {base_topic}?",
                f"Are there limits on {base_topic}?",
                f"What taxes apply to {base_topic}?"
            ])
        elif question_type == 'coverage':
            investigation_queries.extend([
                f"What is excluded from {base_topic}?",
                f"What are the conditions for {base_topic}?",
                f"What documentation is needed for {base_topic}?"
            ])
        
        # Execute investigations in parallel
        investigation_tasks = [
            self._investigate_specific(query) for query in investigation_queries
        ]
        investigation_results = await asyncio.gather(*investigation_tasks)
        
        # Organize findings
        for query, result in zip(investigation_queries, investigation_results):
            if result and result != "No relevant information found in the document.":
                # Categorize the finding
                if any(word in query.lower() for word in ['exception', 'except', 'unless']):
                    findings['exceptions'].append(result)
                elif any(word in query.lower() for word in ['condition', 'if', 'when', 'require']):
                    findings['conditions'].append(result)
                elif any(word in query.lower() for word in ['limit', 'maximum', 'minimum']):
                    findings['limitations'].append(result)
                elif any(word in query.lower() for word in ['document', 'proof', 'submit']):
                    findings['prerequisites'].append(result)
                else:
                    findings['related_info'].append(result)
        
        # Remove duplicates
        for key in findings:
            findings[key] = list(set(findings[key]))
        
        return findings

    async def _investigate_specific(self, query: str) -> str:
        """Execute a specific investigation query."""
        try:
            # Use a lighter search for investigation queries
            search_results = self.vector_store.search(query, k=5)
            if not search_results:
                return ""
            
            # Get just the most relevant chunk
            top_chunk = search_results[0][0] if search_results else ""
            
            # Quick extraction of relevant sentences
            sentences = top_chunk.split('.')
            relevant_sentences = [
                s.strip() for s in sentences 
                if any(word in s.lower() for word in query.lower().split())
            ]
            
            return '. '.join(relevant_sentences[:2]) if relevant_sentences else ""
        except Exception as e:
            logger.warning(f"Investigation query failed: {e}")
            return ""

    async def _find_contradictions(self, question: str, findings: Dict) -> List[str]:
        """Detect contradictions in the findings."""
        contradictions = []
        
        # Look for conflicting numbers
        all_text = str(findings)
        numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:days?|months?|years?|%|percent)', all_text)
        
        if len(set(numbers)) > 1:
            contradictions.append(f"Found different values mentioned: {', '.join(set(numbers))}")
        
        # Look for conflicting terms
        conflict_pairs = [
            ('included', 'excluded'),
            ('covered', 'not covered'),
            ('eligible', 'not eligible'),
            ('allowed', 'prohibited')
        ]
        
        for term1, term2 in conflict_pairs:
            if term1 in all_text.lower() and term2 in all_text.lower():
                contradictions.append(f"Document contains both '{term1}' and '{term2}' references")
        
        return contradictions

    async def _investigate_edge_cases(self, question: str, question_type: str) -> List[str]:
        """Find edge cases and special scenarios."""
        edge_cases = []
        base_topic = self._extract_main_topic(question)
        
        edge_case_queries = [
            f"special cases for {base_topic}",
            f"exceptions to {base_topic}",
            f"unusual circumstances {base_topic}",
            f"emergency situations {base_topic}"
        ]
        
        for query in edge_case_queries:
            result = await self._investigate_specific(query)
            if result and len(result) > 20:
                edge_cases.append(result)
        
        return list(set(edge_cases))[:3]  # Limit to top 3 unique edge cases

    def _extract_main_topic(self, question: str) -> str:
        """Extract the main topic from the question."""
        # Remove question words
        question_words = ['what', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 'could', 'would', 'should']
        words = question.lower().split()
        topic_words = [w for w in words if w not in question_words and len(w) > 3]
        
        # Return the most substantial part
        return ' '.join(topic_words[:3]) if topic_words else question[:30]

    async def _create_detective_report(self, question: str, basic_answer: str,
                                      findings: Dict, contradictions: List[str],
                                      edge_cases: List[str]) -> str:
        """Create a comprehensive detective-style report."""
        
        report_parts = []
        
        # 1. Main Answer
        report_parts.append(f"📋 **DIRECT ANSWER:**\n{basic_answer}")
        
        # 2. Important Exceptions (if any)
        if findings['exceptions']:
            report_parts.append("\n⚠️ **IMPORTANT EXCEPTIONS:**")
            for exception in findings['exceptions'][:3]:
                if exception:
                    report_parts.append(f"• {exception}")
        
        # 3. Conditions and Prerequisites
        if findings['conditions'] or findings['prerequisites']:
            report_parts.append("\n📌 **CONDITIONS & REQUIREMENTS:**")
            for condition in (findings['conditions'] + findings['prerequisites'])[:3]:
                if condition:
                    report_parts.append(f"• {condition}")
        
        # 4. Limitations
        if findings['limitations']:
            report_parts.append("\n🚫 **LIMITATIONS:**")
            for limitation in findings['limitations'][:2]:
                if limitation:
                    report_parts.append(f"• {limitation}")
        
        # 5. Contradictions Alert
        if contradictions:
            report_parts.append("\n⚡ **ATTENTION - INCONSISTENCIES FOUND:**")
            for contradiction in contradictions[:2]:
                report_parts.append(f"• {contradiction}")
            report_parts.append("• Recommend verifying with source document")
        
        # 6. Edge Cases
        if edge_cases:
            report_parts.append("\n🔍 **SPECIAL CASES TO NOTE:**")
            for case in edge_cases[:2]:
                if case:
                    report_parts.append(f"• {case}")
        
        # 7. Additional Insights
        if findings['related_info']:
            report_parts.append("\n💡 **RELATED INFORMATION YOU SHOULD KNOW:**")
            for info in findings['related_info'][:3]:
                if info and len(info) > 20:
                    report_parts.append(f"• {info}")
        
        # Combine all parts
        final_report = '\n'.join(report_parts)
        
        # If the investigation found significant additional information, add a summary
        if len(report_parts) > 2:
            logger.info(f"🎯 Detective found {len(report_parts)-1} additional insight categories")
        
        return final_report

    async def synthesize_answer(self, original_question: str, sub_answers: List[str]) -> str:
        """Enhanced synthesis that maintains detective findings."""
        
        # Create investigation context
        context = "\n\n".join(
            f"Investigation Finding {i+1}:\n{answer}"
            for i, answer in enumerate(sub_answers)
        )
        
        synthesis_prompt = f"""
        You are a detective AI that has investigated multiple aspects of a question.
        Synthesize your findings into a comprehensive answer that includes:
        1. The direct answer
        2. Any important exceptions or conditions
        3. Related information the user should know
        4. Any contradictions or points needing attention
        
        Original Question: {original_question}
        
        Investigation Findings:
        {context}
        
        Provide a detective-style comprehensive answer:
        """
        
        model = self.rag_pipeline.llm_precise
        response = await model.generate_content_async(synthesis_prompt)
        
        return response.text.strip()