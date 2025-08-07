# app/core/answer_validator.py - Answer validation and correction
import re
from typing import Dict, List, Any
import logging
# At the top of answer_validator.py
from rapidfuzz import fuzz
import numpy as np

logger = logging.getLogger(__name__)

class AnswerValidator:
    """Validates and corrects answers"""
    
    def validate(self, question: str, answer: str, question_info: Dict, 
                chunks: List[str]) -> Dict[str, Any]:
        """Validate answer and correct if needed"""
        
        validation_result = {
            'answer': answer,
            'validation_passed': True,
            'corrections': [],
            'sources': [],
            'notes': ''
        }
        
        # Type-specific validation
        if question_info['type'] == 'numerical':
            validation_result = self._validate_numerical(
                question, answer, chunks, validation_result
            )
        elif question_info['type'] == 'yes_no':
            validation_result = self._validate_yes_no(
                answer, validation_result
            )
        elif question_info['type'] == 'list':
            validation_result = self._validate_list(
                question, answer, chunks, validation_result
            )
        
        # General validations
        validation_result = self._validate_completeness(
            answer, question_info, validation_result
        )
        
        validation_result = self._validate_consistency(
            answer, chunks, validation_result
        )
        
        return validation_result
    
    def _validate_numerical(self, question: str, answer: str, chunks: List[str],
                           result: Dict) -> Dict:
        """Validate numerical answers"""
        
        # Extract numbers from answer
        answer_numbers = re.findall(r'[-+]?\d*\.?\d+', answer)
        
        if not answer_numbers:
            result['validation_passed'] = False
            result['notes'] = 'No numerical value found in answer'
            return result
        
        # Check if numbers appear in context
        context_text = ' '.join(chunks[:5])
        context_numbers = re.findall(r'[-+]?\d*\.?\d+', context_text)
        
        # Verify calculation if needed
        if 'calculate' in question.lower() or 'total' in question.lower():
            # Try to verify the calculation
            try:
                # Extract operation
                if 'sum' in question.lower() or 'total' in question.lower():
                    numbers = [float(n) for n in context_numbers]
                    expected = sum(numbers)
                    actual = float(answer_numbers[0])
                    
                    if abs(expected - actual) > 0.01:
                        result['corrections'].append(
                            f"Calculation verification: expected {expected}, got {actual}"
                        )
                        result['answer'] = f"The correct total is {expected}"
            except:
                pass
        
        return result
    
    def _validate_yes_no(self, answer: str, result: Dict) -> Dict:
        """Validate yes/no answers"""
        
        answer_lower = answer.lower()
        
        # Check if answer starts with yes/no
        if not (answer_lower.startswith('yes') or answer_lower.startswith('no')):
            # Try to infer from answer
            if 'not' in answer_lower or 'cannot' in answer_lower or "don't" in answer_lower:
                result['answer'] = f"No. {answer}"
            elif 'can' in answer_lower or 'is' in answer_lower:
                result['answer'] = f"Yes. {answer}"
            else:
                result['validation_passed'] = False
                result['notes'] = 'Yes/No answer not clearly stated'
        
        return result
    
    def _validate_list(self, question: str, answer: str, chunks: List[str],
                      result: Dict) -> Dict:
        """Validate list answers"""
        
        # Check if answer contains a list structure
        list_indicators = [
            r'^\d+\.',  # Numbered list
            r'^-',      # Bullet points
            r'^•',      # Bullet points
            r'^\*',     # Asterisk points
        ]
        
        lines = answer.split('\n')
        has_list = any(
            any(re.match(pattern, line.strip()) for pattern in list_indicators)
           for line in lines if line.strip()
        )
       
        if not has_list and 'list' in question.lower():
            # Try to format as list
            items = re.split(r'[,;]|\band\b', answer)
            if len(items) > 1:
                formatted_list = '\n'.join(f"• {item.strip()}" for item in items if item.strip())
                result['answer'] = formatted_list
                result['corrections'].append('Reformatted answer as list')
        
        # Check completeness
        if 'all' in question.lower():
            context_text = ' '.join(chunks[:10])
            
            # Look for enumeration patterns in context
            enum_patterns = [
                r'(?:first|second|third|fourth|fifth)',
                r'(?:\d+\)|\d+\.)',
                r'(?:[a-z]\)|\([a-z]\))'
            ]
            
            for pattern in enum_patterns:
                context_items = re.findall(pattern, context_text, re.IGNORECASE)
                answer_items = re.findall(pattern, answer, re.IGNORECASE)
                
                if len(context_items) > len(answer_items):
                    result['notes'] = f'Answer may be incomplete. Found {len(context_items)} items in context but only {len(answer_items)} in answer'
        
        return result
    
    def _validate_completeness(self, answer: str, question_info: Dict,
                                result: Dict) -> Dict:
        """Check if answer is complete"""
        
        # Check minimum length
        if len(answer) < 20:
            result['validation_passed'] = False
            result['notes'] = 'Answer too short'
        
        # Check for incomplete sentences
        if answer.endswith('...') or answer.endswith('etc'):
            result['notes'] = 'Answer appears incomplete'
        
        # Check for error messages
        error_phrases = [
            'unable to', 'cannot find', 'not found',
            'error', 'timeout', 'failed'
        ]
        
        if any(phrase in answer.lower() for phrase in error_phrases):
            result['validation_passed'] = False
            result['notes'] = 'Answer contains error indication'
        
        return result
    
    def _validate_consistency(self, answer: str, chunks: List[str],
                            result: Dict) -> Dict:
        """Check answer consistency with context"""
        
        # Extract key facts from answer
        answer_facts = self._extract_facts(answer)
        
        # Check if facts appear in context
        context_text = ' '.join(chunks[:10]).lower()
        
        unsupported_facts = []
        for fact in answer_facts:
            if fact.lower() not in context_text:
                # Use fuzzy matching for variations
                if not self._fuzzy_match_in_context(fact, context_text):
                    unsupported_facts.append(fact)
        
        if unsupported_facts:
            result['notes'] = f'Some facts may not be supported by context: {unsupported_facts[:3]}'
        
        # Extract supporting chunks
        for i, chunk in enumerate(chunks[:5]):
            if any(fact.lower() in chunk.lower() for fact in answer_facts):
                result['sources'].append({
                    'chunk_index': i,
                    'preview': chunk[:200]
                })
        
        return result
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        facts = []
        
        # Extract quoted text
        quoted = re.findall(r'"([^"]+)"', text)
        facts.extend(quoted)
        
        # Extract numbers with context
        numbers = re.findall(r'(\w+\s+)?(\d+(?:\.\d+)?)\s+(\w+)', text)
        for match in numbers:
            facts.append(' '.join(match).strip())
        
        # Extract proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        facts.extend(proper_nouns)
        
        return facts
    
    def _fuzzy_match_in_context(self, fact: str, context: str) -> bool:
        """Check if fact appears in context with fuzzy matching"""
        from rapidfuzz import fuzz
        
        # Split context into sentences
        sentences = re.split(r'[.!?]', context)
        
        # Check each sentence
        for sentence in sentences:
            if fuzz.partial_ratio(fact.lower(), sentence.lower()) > 80:
                return True
        
        return False