# app/agents/advanced_query_agent.py
import logging
import asyncio
import re
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict
from app.models.query import QueryRequest, QueryResponse
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
        self.investigation_cache = {}  # Cache investigation results
        
        # Detective patterns for different question types
        self.detective_patterns = {
            'time_period': {
                'triggers': ['period', 'days', 'months', 'years', 'time', 'duration', 'deadline', 'when', 'waiting'],
                'investigate': ['exception', 'extension', 'grace period', 'holiday', 'business days', 
                               'calendar days', 'renewal', 'expiry', 'early', 'late', 'penalty', 'reduced', 'extended']
            },
            'money': {
                'triggers': ['cost', 'price', 'fee', 'amount', 'payment', 'premium', 'charge', 'dollar', 'rupee', 
                           'salary', 'compensation', 'reimbursement', 'deductible', 'investment', 'billion'],
                'investigate': ['tax', 'penalty', 'interest', 'late fee', 'discount', 'maximum', 
                               'minimum', 'limit', 'co-pay', 'excluded', 'additional charges', 'hidden costs']
            },
            'coverage': {
                'triggers': ['covered', 'eligible', 'included', 'benefit', 'claim', 'insurance', 'policy'],
                'investigate': ['excluded', 'exception', 'not covered', 'pre-existing', 'waiting period',
                               'maximum benefit', 'sub-limit', 'conditions', 'requirements', 'documentation', 'denial']
            },
            'eligibility': {
                'triggers': ['eligible', 'qualify', 'requirement', 'criteria', 'who can', 'allowed'],
                'investigate': ['age limit', 'documentation', 'proof', 'minimum', 'maximum', 
                               'excluded', 'special cases', 'grandfather clause', 'restrictions']
            },
            'process': {
                'triggers': ['how to', 'process', 'procedure', 'steps', 'apply', 'claim', 'submit', 'get', 'find'],
                'investigate': ['deadline', 'required documents', 'forms', 'approval', 'rejection',
                               'appeal', 'review', 'turnaround time', 'contact', 'prerequisites']
            },
            'location': {
                'triggers': ['where', 'location', 'city', 'landmark', 'gateway', 'tower', 'monument'],
                'investigate': ['wrong location', 'swapped', 'parallel world', 'original location', 'multiple locations']
            },
            'api': {
                'triggers': ['endpoint', 'api', 'url', 'flight', 'token', 'secret'],
                'investigate': ['authentication', 'parameters', 'response', 'error', 'fallback']
            },
            'policy': {
                'triggers': ['tariff', 'policy', 'announcement', 'exemption', 'regulation'],
                'investigate': ['exceptions', 'who is affected', 'implications', 'consequences', 'trade war']
            }
        }

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Main entry point for the detective agent."""
        logger.info(f"🔍 Detective Agent activated for: {request.documents}")
        
        try:
            # Step 1: Get or create vector store
            self.vector_store = await self.rag_pipeline.get_or_create_vector_store(request.documents)
            
            # Step 2: Process questions with detective mindset
            tasks = []
            for q in request.questions:
                tasks.append(self.investigate_question(q))
            
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in answers
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"Error processing question {i}: {answer}")
                    final_answers.append(f"Error processing question: {str(answer)[:200]}")
                else:
                    final_answers.append(answer)
            
            return QueryResponse(answers=final_answers)
            
        except Exception as e:
            logger.error(f"Critical error in detective agent: {e}", exc_info=True)
            error_msg = "Document processing error. Please try again."
            return QueryResponse(answers=[error_msg] * len(request.questions))

    async def investigate_question(self, question: str) -> str:
        """Detective-style investigation that goes beyond the explicit question."""
        try:
            logger.info(f"🕵️ Investigating: '{question[:100]}...'")
            
            # Check cache first
            cache_key = f"{question[:100]}"
            if cache_key in self.investigation_cache:
                logger.info("Using cached investigation result")
                return self.investigation_cache[cache_key]
            
            # Detect question type and patterns
            question_types, keywords = self._detect_question_patterns(question)
            logger.info(f"📊 Patterns: {question_types}, Keywords: {len(keywords)}")
            
            # Phase 1: Get the basic answer
            basic_answer = await self._get_basic_answer(question)
            
            # If basic answer is too short or generic, dig deeper
            if len(basic_answer) < 50 or "no relevant information" in basic_answer.lower():
                basic_answer = await self._deep_search(question)
            
            # Phase 2: Conduct investigation based on patterns
            investigation_findings = await self._conduct_investigation(
                question, question_types, keywords, basic_answer
            )
            
            # Phase 3: Check for contradictions
            contradictions = self._find_contradictions(basic_answer, investigation_findings)
            
            # Phase 4: Find edge cases
            edge_cases = await self._investigate_edge_cases(question, question_types)
            
            # Phase 5: Create comprehensive detective report
            final_answer = self._create_detective_report(
                question, basic_answer, investigation_findings, contradictions, edge_cases
            )
            
            # Clean up encoding issues in the final answer
            final_answer = self._clean_text(final_answer)
            
            # Cache the result
            self.investigation_cache[cache_key] = final_answer
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Investigation failed for question: {e}", exc_info=True)
            return f"Investigation error: {str(e)[:200]}"

    def _detect_question_patterns(self, question: str) -> Tuple[List[str], List[str]]:
        """Detect multiple question types and extract key investigation terms."""
        question_lower = question.lower()
        detected_patterns = []
        all_keywords = set()
        
        # Check all patterns
        for pattern_type, pattern_info in self.detective_patterns.items():
            for trigger in pattern_info['triggers']:
                if trigger in question_lower:
                    detected_patterns.append(pattern_type)
                    all_keywords.update(pattern_info['investigate'])
                    break
        
        # Default to general if no pattern detected
        if not detected_patterns:
            detected_patterns = ['general']
            all_keywords = {'exception', 'condition', 'requirement', 'limit', 'special case', 
                           'important', 'note', 'warning', 'caution', 'additional'}
        
        return detected_patterns, list(all_keywords)[:15]  # Limit keywords

    async def _get_basic_answer(self, question: str) -> str:
        """Get the straightforward answer to the question."""
        try:
            logger.info("📝 Getting basic answer...")
            answer = await self.rag_pipeline.answer_question(question, self.vector_store)
            return answer if answer else "No direct answer found."
        except Exception as e:
            logger.error(f"Basic answer failed: {e}")
            return "Unable to generate basic answer."

    async def _deep_search(self, question: str) -> str:
        """Perform deeper search when basic answer is insufficient."""
        logger.info("🔬 Performing deep search...")
        
        # Extract key terms from question
        key_terms = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b|\b[a-z]{4,}\b', question)
        
        # Search with different query formulations
        search_queries = [
            question,
            ' '.join(key_terms),
            question.replace('?', ''),
            f"information about {' '.join(key_terms[:3])}"
        ]
        
        all_results = []
        for query in search_queries[:3]:  # Limit searches
            try:
                results = self.vector_store.search(query, k=5)
                all_results.extend([r[0] for r in results[:2]])
            except:
                continue
        
        if all_results:
            # Combine unique results
            unique_results = list(dict.fromkeys(all_results))[:3]
            context = '\n'.join(unique_results)
            
            # Generate answer from expanded context
            try:
                answer = await self.rag_pipeline._generate_answer(question, unique_results, True)
                return answer
            except:
                return unique_results[0] if unique_results else "No information found."
        
        return "No relevant information found after deep search."

    async def _conduct_investigation(self, question: str, question_types: List[str], 
                                    keywords: List[str], basic_answer: str) -> Dict[str, List[str]]:
        """Conduct detailed investigation based on question patterns."""
        findings = {
            'exceptions': [],
            'conditions': [],
            'related_info': [],
            'important_notes': [],
            'prerequisites': [],
            'limitations': [],
            'warnings': []
        }
        
        # Extract main topic
        base_topic = self._extract_main_topic(question)
        
        # Create investigation queries
        investigation_queries = []
        
        # Type-specific investigations
        for q_type in question_types[:2]:  # Limit to top 2 types
            if q_type == 'time_period':
                investigation_queries.extend([
                    f"{base_topic} exceptions",
                    f"{base_topic} extensions",
                    f"{base_topic} penalties"
                ])
            elif q_type == 'money':
                investigation_queries.extend([
                    f"{base_topic} additional costs",
                    f"{base_topic} limits",
                    f"{base_topic} exemptions"
                ])
            elif q_type == 'location':
                investigation_queries.extend([
                    f"{base_topic} multiple locations",
                    f"{base_topic} appears twice",
                    f"{base_topic} contradiction"
                ])
        
        # Keyword-based investigations
        for keyword in keywords[:5]:
            investigation_queries.append(f"{base_topic} {keyword}")
        
        # Execute investigations concurrently
        investigation_tasks = []
        for query in investigation_queries[:10]:  # Limit total investigations
            investigation_tasks.append(self._investigate_specific(query))
        
        investigation_results = await asyncio.gather(*investigation_tasks, return_exceptions=True)
        
        # Process and categorize findings
        for query, result in zip(investigation_queries, investigation_results):
            if isinstance(result, Exception):
                continue
                
            if result and len(result) > 20 and result != basic_answer:
                # Categorize based on content
                result_lower = result.lower()
                
                if any(word in result_lower for word in ['except', 'unless', 'but not']):
                    findings['exceptions'].append(result[:200])
                elif any(word in result_lower for word in ['if', 'when', 'require', 'must']):
                    findings['conditions'].append(result[:200])
                elif any(word in result_lower for word in ['limit', 'maximum', 'minimum', 'up to']):
                    findings['limitations'].append(result[:200])
                elif any(word in result_lower for word in ['warning', 'caution', 'important', 'note']):
                    findings['warnings'].append(result[:200])
                elif any(word in result_lower for word in ['document', 'proof', 'submit', 'form']):
                    findings['prerequisites'].append(result[:200])
                else:
                    findings['related_info'].append(result[:200])
        
        # Remove duplicates and limit each category
        for key in findings:
            unique_items = []
            seen = set()
            for item in findings[key]:
                item_key = item[:50]  # Use first 50 chars as key
                if item_key not in seen:
                    seen.add(item_key)
                    unique_items.append(item)
            findings[key] = unique_items[:3]  # Max 3 items per category
        
        return findings

    async def _investigate_specific(self, query: str) -> str:
        """Execute a specific investigation query."""
        try:
            # Search for specific information
            search_results = self.vector_store.search(query, k=3)
            
            if not search_results:
                return ""
            
            # Extract relevant content
            relevant_content = []
            for chunk, score, _ in search_results:
                if score > 0.1:  # Relevance threshold
                    # Extract sentences containing query terms
                    sentences = chunk.split('.')
                    query_terms = query.lower().split()
                    
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        if any(term in sentence_lower for term in query_terms if len(term) > 3):
                            relevant_content.append(sentence.strip())
                            if len(relevant_content) >= 2:
                                break
                
                if len(relevant_content) >= 2:
                    break
            
            return '. '.join(relevant_content[:2]) if relevant_content else ""
            
        except Exception as e:
            logger.warning(f"Specific investigation failed: {e}")
            return ""

    def _find_contradictions(self, basic_answer: str, findings: Dict) -> List[str]:
        """Detect contradictions and inconsistencies."""
        contradictions = []
        
        # Combine all text for analysis
        all_text = basic_answer + ' ' + ' '.join(
            ' '.join(items) for items in findings.values() if items
        )
        
        # Find conflicting numbers
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(days?|months?|years?|dollars?|%|percent)', all_text)
        
        if numbers:
            # Group by unit
            number_groups = defaultdict(list)
            for num, unit in numbers:
                number_groups[unit.lower()].append(float(num))
            
            # Check for different values with same unit
            for unit, values in number_groups.items():
                unique_values = list(set(values))
                if len(unique_values) > 1:
                    contradictions.append(
                        f"Conflicting {unit}: {', '.join(map(str, unique_values))}"
                    )
        
        # Find conflicting terms
        conflict_pairs = [
            ('included', 'excluded'),
            ('covered', 'not covered'),
            ('eligible', 'not eligible'),
            ('allowed', 'prohibited'),
            ('yes', 'no')
        ]
        
        text_lower = all_text.lower()
        for term1, term2 in conflict_pairs:
            if term1 in text_lower and term2 in text_lower:
                contradictions.append(f"Both '{term1}' and '{term2}' mentioned")
        
        # Check for duplicate landmarks (specific to parallel world puzzle)
        landmark_pattern = r'(Big Ben|Taj Mahal|Gateway of India|Eiffel Tower)'
        landmarks = re.findall(landmark_pattern, all_text, re.IGNORECASE)
        
        if landmarks:
            landmark_counts = defaultdict(int)
            for landmark in landmarks:
                landmark_counts[landmark.lower()] += 1
            
            for landmark, count in landmark_counts.items():
                if count > 1:
                    contradictions.append(f"'{landmark}' appears {count} times - verify correct location")
        
        return contradictions[:3]  # Limit contradictions

    async def _investigate_edge_cases(self, question: str, question_types: List[str]) -> List[str]:
        """Find edge cases and special scenarios."""
        edge_cases = []
        base_topic = self._extract_main_topic(question)
        
        # Type-specific edge case queries
        edge_queries = []
        
        if 'location' in question_types:
            edge_queries.extend([
                f"{base_topic} multiple entries",
                f"{base_topic} duplicate",
                f"{base_topic} ambiguous"
            ])
        elif 'process' in question_types:
            edge_queries.extend([
                f"{base_topic} failure",
                f"{base_topic} error",
                f"{base_topic} alternative"
            ])
        else:
            edge_queries.extend([
                f"{base_topic} special case",
                f"{base_topic} exception",
                f"{base_topic} unusual"
            ])
        
        # Search for edge cases
        for query in edge_queries[:3]:
            result = await self._investigate_specific(query)
            if result and len(result) > 30:
                edge_cases.append(result)
        
        return list(dict.fromkeys(edge_cases))[:2]  # Unique edge cases

    def _extract_main_topic(self, question: str) -> str:
        """Extract the main topic from the question."""
        # Remove question words and punctuation
        question_words = {'what', 'when', 'where', 'why', 'how', 'is', 'are', 'can', 
                         'could', 'would', 'should', 'does', 'do', 'if', 'the', 'a', 'an'}
        
        words = question.lower().replace('?', '').split()
        
        # Find important words (longer words, capitalized words, numbers)
        important_words = []
        
        for word in question.split():
            word_clean = word.strip('?,.')
            if (word_clean and 
                word_clean.lower() not in question_words and 
                (len(word_clean) > 3 or word_clean[0].isupper() or word_clean.isdigit())):
                important_words.append(word_clean)
        
        # Return first few important words
        return ' '.join(important_words[:3]) if important_words else question[:30]

    def _create_detective_report(self, question: str, basic_answer: str,
                                findings: Dict, contradictions: List[str],
                                edge_cases: List[str]) -> str:
        """Create a comprehensive detective-style report."""
        
        report_parts = []
        
        # 1. Direct Answer
        report_parts.append(f"📋 **DIRECT ANSWER:**")
        report_parts.append(basic_answer)
        
        # 2. Important Exceptions
        if findings.get('exceptions'):
            report_parts.append("\n⚠️ **IMPORTANT EXCEPTIONS:**")
            for exception in findings['exceptions']:
                clean_exception = self._clean_text(exception)
                if clean_exception:
                    report_parts.append(f"• {clean_exception}")
        
        # 3. Conditions & Requirements
        conditions_found = findings.get('conditions', []) + findings.get('prerequisites', [])
        if conditions_found:
            report_parts.append("\n📌 **CONDITIONS & REQUIREMENTS:**")
            for condition in conditions_found[:3]:
                clean_condition = self._clean_text(condition)
                if clean_condition:
                    report_parts.append(f"• {clean_condition}")
        
        # 4. Limitations
        if findings.get('limitations'):
            report_parts.append("\n🚫 **LIMITATIONS:**")
            for limitation in findings['limitations']:
                clean_limit = self._clean_text(limitation)
                if clean_limit:
                    report_parts.append(f"• {clean_limit}")
        
        # 5. Contradictions Alert
        if contradictions:
            report_parts.append("\n⚡ **ATTENTION - INCONSISTENCIES FOUND:**")
            for contradiction in contradictions:
                report_parts.append(f"• {contradiction}")
            report_parts.append("• Verify with source document for accuracy")
        
        # 6. Special Cases
        if edge_cases:
            report_parts.append("\n🔍 **SPECIAL CASES TO NOTE:**")
            for case in edge_cases:
                clean_case = self._clean_text(case)
                if clean_case:
                    report_parts.append(f"• {clean_case}")
        
        # 7. Warnings
        if findings.get('warnings'):
            report_parts.append("\n⚠️ **WARNINGS:**")
            for warning in findings['warnings']:
                clean_warning = self._clean_text(warning)
                if clean_warning:
                    report_parts.append(f"• {clean_warning}")
        
        # 8. Additional Information
        if findings.get('related_info'):
            report_parts.append("\n💡 **RELATED INFORMATION YOU SHOULD KNOW:**")
            for info in findings['related_info'][:3]:
                clean_info = self._clean_text(info)
                if clean_info and len(clean_info) > 20:
                    report_parts.append(f"• {clean_info}")
        
        # Build final report
        final_report = '\n'.join(report_parts)
        
        # If investigation found significant insights, log it
        total_findings = sum(len(v) for v in findings.values())
        if total_findings > 0:
            logger.info(f"🎯 Detective found {total_findings} insights beyond the basic answer")
        
        return final_report

    def _clean_text(self, text: str) -> str:
        """Clean text from encoding issues and artifacts."""
        if not text:
            return ""
        
        # Remove common encoding artifacts
        text = re.sub(r'\(cid:\d+\)', '', text)  # Remove (cid:X) artifacts
        text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)  # Remove unicode escapes
        text = re.sub(r'\x00+', ' ', text)  # Remove null bytes
        text = re.sub(r'[^\x20-\x7E\n\r\t]+', '', text)  # Keep only printable ASCII
        
        # Clean up excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove incomplete sentences at the end
        if text and not text[-1] in '.!?':
            # Find last complete sentence
            last_period = text.rfind('.')
            if last_period > len(text) // 2:  # Only truncate if we keep most of the text
                text = text[:last_period + 1]
        
        return text.strip()