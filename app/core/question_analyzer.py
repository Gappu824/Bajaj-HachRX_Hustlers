# app/core/question_analyzer.py - Intelligent question analysis
import re
from typing import Dict, List, Any
import nltk
from textstat import flesch_reading_ease
# Add at the top of question_analyzer.py after imports
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Fix flesch_reading_ease import
try:
    from textstat import flesch_reading_ease
except ImportError:
    # Fallback if textstat not available
    def flesch_reading_ease(text):
        return 50  # Default middle complexity
class QuestionAnalyzer:
    """Analyzes questions to determine type and complexity"""
    
    def __init__(self):
        # Question type patterns
        self.patterns = {
            'numerical': [
                r'\b(how many|how much|calculate|total|sum|average|mean|median)\b',
                r'\b(percentage|percent|ratio|proportion)\b',
                r'\b(count|number of|amount)\b'
            ],
            'yes_no': [
                r'^(is|are|was|were|do|does|did|can|could|should|would|will)\b',
                r'\b(true|false|correct|incorrect)\b'
            ],
            'list': [
                r'\b(list|enumerate|name all|what are all|identify all)\b',
                r'\b(types|kinds|categories|examples)\b'
            ],
            'comparison': [
                r'\b(compare|contrast|difference|similar|versus|vs)\b',
                r'\b(better|worse|more|less|greater|smaller)\b'
            ],
            'definition': [
                r'\b(what is|what are|define|explain|describe)\b',
                r'\b(meaning|definition|explanation)\b'
            ],
            'process': [
                r'\b(how to|steps|procedure|process|method)\b',
                r'\b(workflow|sequence|order)\b'
            ],
            'causal': [
                r'\b(why|because|cause|reason|result|effect)\b',
                r'\b(lead to|due to|therefore|consequently)\b'
            ]
        }
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """Comprehensive question analysis"""
        
        question_lower = question.lower()
        
        # Detect question type
        question_type = self._detect_type(question_lower)
        
        # Check complexity
        complexity = self._assess_complexity(question)
        
        # Extract entities
        entities = self._extract_entities(question)
        
        # Determine if multi-step reasoning needed
        requires_multi_step = self._needs_multi_step(question, question_type, complexity)
        
        # Check if high precision needed
        requires_precision = self._needs_precision(question_type, question_lower)
        
        return {
            'type': question_type,
            'complexity': complexity,
            'entities': entities,
            'requires_multi_step': requires_multi_step,
            'requires_precision': requires_precision,
            'main_term': self._extract_main_term(question)
        }
    
    def _detect_type(self, question: str) -> str:
        """Detect question type using patterns"""
        
        scores = {}
        for q_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns 
                       if re.search(pattern, question, re.IGNORECASE))
            scores[q_type] = score
        
        # Get type with highest score
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        # Check for multi-part questions
        if ' and ' in question or ',' in question:
            return 'multi_part'
        
        return 'general'
    
    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity"""
        
        # Factors for complexity
        word_count = len(question.split())
        clause_count = len(re.findall(r'[,;]|and|or|but', question))
        
        # Reading ease score (lower = more complex)
        try:
            reading_ease = flesch_reading_ease(question)
        except:
            reading_ease = 50
        
        # Calculate complexity score
        complexity_score = 0
        
        if word_count > 20:
            complexity_score += 1
        if word_count > 30:
            complexity_score += 1
        
        if clause_count > 2:
            complexity_score += 1
        
        if reading_ease < 30:
            complexity_score += 2
        elif reading_ease < 50:
            complexity_score += 1
        
        # Map to complexity level
        if complexity_score >= 4:
            return 'complex'
        elif complexity_score >= 2:
            return 'moderate'
        else:
            return 'simple'
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract key entities from question"""
        
        entities = []
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]*)"', question)
        entities.extend(quoted)
        
        # Extract capitalized words (likely proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        entities.extend(capitalized)
        
        # Extract numbers with units
        numbers = re.findall(r'\b\d+(?:\.\d+)?\s*\w+\b', question)
        entities.extend(numbers)
        
        return list(set(entities))
    
    def _needs_multi_step(self, question: str, q_type: str, complexity: str) -> bool:
        """Determine if multi-step reasoning is needed"""
        
        if complexity == 'complex':
            return True
        
        if q_type in ['comparison', 'multi_part', 'causal']:
            return True
        
        # Check for multiple operations
        if 'then' in question.lower() or 'after' in question.lower():
            return True
        
        # Check for calculations with multiple steps
        if q_type == 'numerical' and ('and' in question or 'total' in question):
            return True
        
        return False
    
    def _needs_precision(self, q_type: str, question: str) -> bool:
        """Determine if high precision is needed"""
        
        if q_type in ['numerical', 'list']:
            return True
        
        precision_keywords = [
            'exact', 'precise', 'specific', 'all', 'every',
            'complete', 'full', 'detailed', 'calculate'
        ]
        
        return any(keyword in question for keyword in precision_keywords)
    
    def _extract_main_term(self, question: str) -> str:
        """Extract the main term or concept being asked about"""
        
        # Remove question words
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        words = question.lower().split()
        
        # Filter out question words and common words
        filtered = [w for w in words if w not in question_words and len(w) > 3]
        
        # Return the first significant noun/term
        if filtered:
            return filtered[0]
        
        return ""