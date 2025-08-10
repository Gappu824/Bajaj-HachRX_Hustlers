#!/usr/bin/env python3
"""
Quick accuracy test for Malayalam improvements
"""

import time
import logging
from app.agents.advanced_query_agent import AdvancedQueryAgent
from app.core.rag_pipeline import HybridRAGPipeline
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_language_detection():
    """Test the enhanced language detection"""
    logger.info("🔍 Testing language detection...")
    
    # Create a minimal agent instance for testing
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Faster model
    rag_pipeline = HybridRAGPipeline(embedding_model)
    agent = AdvancedQueryAgent(rag_pipeline)
    
    # Test Malayalam questions
    malayalam_questions = [
        "ട്രംപ് ഏത് ദിവസമാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്?",
        "ഏത് ഉത്പന്നങ്ങൾക്ക് ഈ 100% ഇറക്കുമതി ശുൽകം ബാധകമാണ്?",
        "ഏത് സാഹചര്യത്തിൽ ഒരു കമ്പനിയ്ക്ക് ഈ 100% ശുൽകത്തിൽ നിന്നും നിന്നും ഒഴികെയാക്കും?",
        "വിദേശ ആശ്രിതത്വം കുറയ്ക്കാനുള്ള തന്ത്രം എന്താണ്?"
    ]
    
    english_questions = [
        "What was Apple's investment commitment?",
        "What impact will this policy have?",
        "Who is exempted from the tariff?"
    ]
    
    # Test language detection
    for i, question in enumerate(malayalam_questions):
        detected = agent._detect_language(question)
        logger.info(f"Malayalam Q{i+1}: {detected} - {question[:30]}...")
        if detected != "malayalam":
            logger.warning(f"⚠️  Malayalam question {i+1} not detected correctly")
    
    for i, question in enumerate(english_questions):
        detected = agent._detect_language(question)
        logger.info(f"English Q{i+1}: {detected} - {question[:30]}...")
        if detected != "english":
            logger.warning(f"⚠️  English question {i+1} not detected correctly")

def test_malayalam_patterns():
    """Test Malayalam pattern detection"""
    logger.info("🔍 Testing Malayalam pattern detection...")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    rag_pipeline = HybridRAGPipeline(embedding_model)
    agent = AdvancedQueryAgent(rag_pipeline)
    
    test_questions = [
        "ട്രംപ് ഏത് ദിവസമാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്?",
        "ഏത് ഉത്പന്നങ്ങൾക്ക് ഈ 100% ഇറക്കുമതി ശുൽകം ബാധകമാണ്?",
        "ഏത് സാഹചര്യത്തിൽ ഒരു കമ്പനിയ്ക്ക് ഈ 100% ശുൽകത്തിൽ നിന്നും നിന്നും ഒഴികെയാക്കും?",
        "വിദേശ ആശ്രിതത്വം കുറയ്ക്കാനുള്ള തന്ത്രം എന്താണ്?"
    ]
    
    for i, question in enumerate(test_questions):
        pattern = agent._detect_malayalam_question_pattern(question)
        logger.info(f"Q{i+1} Pattern: {pattern} - {question[:40]}...")

def test_keyword_extraction():
    """Test Malayalam keyword extraction"""
    logger.info("🔍 Testing keyword extraction...")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    rag_pipeline = HybridRAGPipeline(embedding_model)
    agent = AdvancedQueryAgent(rag_pipeline)
    
    test_question = "ട്രംപ് ഏത് ദിവസമാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്?"
    keywords = agent._extract_malayalam_keywords(test_question)
    logger.info(f"Keywords extracted: {keywords}")
    
    enhanced = agent._enhance_malayalam_query(test_question)
    logger.info(f"Enhanced query: {enhanced}")

def test_response_cleaning():
    """Test response cleaning functionality"""
    logger.info("🔍 Testing response cleaning...")
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    rag_pipeline = HybridRAGPipeline(embedding_model)
    agent = AdvancedQueryAgent(rag_pipeline)
    
    test_responses = [
        "🔸 Here is the answer with emojis ✨ and icons 💡",
        "This is a clean response without any unwanted characters",
        "ട്രംപ് ഏത് ദിവസമാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്? 🔸 ഇതാണ് ഉത്തരം ✨"
    ]
    
    for i, response in enumerate(test_responses):
        cleaned = agent._clean_response(response)
        logger.info(f"Response {i+1}:")
        logger.info(f"  Original: {response}")
        logger.info(f"  Cleaned: {cleaned}")
        
        # Check for unwanted characters
        unwanted_chars = ['🔸', '🔹', '⚡', '✨', '💡', '📝', '📌', '📍', '🎯', '✅', '❌', '⚠️', '🚨', '💯', '🔥', '💪', '🙏', '🤝', '👋', '👌', '👍', '👎']
        has_unwanted = any(char in cleaned for char in unwanted_chars)
        if has_unwanted:
            logger.warning(f"⚠️  Response {i+1} still contains unwanted characters")
        else:
            logger.info(f"✅ Response {i+1} is clean")

def main():
    """Run all quick tests"""
    logger.info("🚀 Starting quick accuracy tests...")
    start_time = time.time()
    
    try:
        test_language_detection()
        test_malayalam_patterns()
        test_keyword_extraction()
        test_response_cleaning()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ All tests completed in {elapsed:.2f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
