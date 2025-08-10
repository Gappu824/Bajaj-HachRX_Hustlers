#!/usr/bin/env python3
"""
Test script for Malayalam accuracy improvements
"""

import asyncio
import time
import logging
from app.main import app
from app.models.query import QueryRequest
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_malayalam_accuracy():
    """Test the Malayalam accuracy improvements"""
    
    client = TestClient(app)
    
    # Test with the specific Malayalam questions from user feedback
    test_request = QueryRequest(
        documents="https://hackrx.blob.core.windows.net/hackrx/rounds/News.pdf?sv=2023-01-03&spr=https&st=2025-08-07T17%3A10%3A11Z&se=2026-08-08T17%3A10%3A00Z&sr=b&sp=r&sig=ybRsnfv%2B6VbxPz5xF7kLLjC4ehU0NF7KDkXua9ujSf0%3D",
        questions=[
            "ട്രംപ് ഏത് ദിവസമാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്?",
            "ഏത് ഉത്പന്നങ്ങൾക്ക് ഈ 100% ഇറക്കുമതി ശുൽകം ബാധകമാണ്?",
            "ഏത് സാഹചര്യത്തിൽ ഒരു കമ്പനിയ്ക്ക് ഈ 100% ശുൽകത്തിൽ നിന്നും നിന്നും ഒഴികെയാക്കും?",
            "വിദേശ ആശ്രിതത്വം കുറയ്ക്കാനുള്ള തന്ത്രം എന്താണ്?",
            "What was Apple's investment commitment and what was its objective?",
            "What impact will this new policy have on consumers and the global market?",
            "Identify the two main goals of the new US tariff policy.",
            "Which specific company is mentioned in relation to a major investment, and what is the amount?",
            "Summarize the new policy in one sentence in English.",
            "Who is exempted from the 100% tariff and why?",
            "What could trigger trade war reactions?",
            "Find all policy implications and consequences mentioned in the document."
        ]
    )
    
    logger.info("🚀 Starting Malayalam accuracy test...")
    
    start_time = time.time()
    
    response = client.post("/api/v1/hackrx/run", json=test_request.model_dump())
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"⏱️  Response time: {elapsed:.2f} seconds")
    logger.info(f"📝 Response status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        answers = data.get('answers', [])
        logger.info(f"✅ Generated {len(answers)} answers")
        
        # Analyze each answer
        malayalam_questions = test_request.questions[:4]  # First 4 are Malayalam
        english_questions = test_request.questions[4:]    # Rest are English
        
        logger.info("\n📊 MALAYALAM QUESTIONS ANALYSIS:")
        for i, (question, answer) in enumerate(zip(malayalam_questions, answers[:4])):
            logger.info(f"\nQuestion {i+1}: {question}")
            logger.info(f"Answer: {answer}")
            
            # Check if answer is in Malayalam
            malayalam_chars = len([c for c in answer if '\u0d00' <= c <= '\u0d7f'])
            malayalam_ratio = malayalam_chars / len(answer) if answer else 0
            
            if malayalam_ratio > 0.1:  # More than 10% Malayalam characters
                logger.info("✅ Answer is in Malayalam")
            else:
                logger.warning("⚠️  Answer is not in Malayalam")
            
            # Check for unwanted characters
            unwanted_chars = ['🔸', '🔹', '⚡', '✨', '💡', '📝', '📌', '📍', '🎯', '✅', '❌', '⚠️', '🚨', '💯', '🔥', '💪', '🙏', '🤝', '👋', '👌', '👍', '👎']
            has_unwanted = any(char in answer for char in unwanted_chars)
            if has_unwanted:
                logger.warning("⚠️  Answer contains unwanted characters/icons")
            else:
                logger.info("✅ Answer is clean (no unwanted characters)")
        
        logger.info("\n📊 ENGLISH QUESTIONS ANALYSIS:")
        for i, (question, answer) in enumerate(zip(english_questions, answers[4:])):
            logger.info(f"\nQuestion {i+5}: {question}")
            logger.info(f"Answer: {answer}")
            
            # Check for unwanted characters
            unwanted_chars = ['🔸', '🔹', '⚡', '✨', '💡', '📝', '📌', '📍', '🎯', '✅', '❌', '⚠️', '🚨', '💯', '🔥', '💪', '🙏', '🤝', '👋', '👌', '👍', '👎']
            has_unwanted = any(char in answer for char in unwanted_chars)
            if has_unwanted:
                logger.warning("⚠️  Answer contains unwanted characters/icons")
            else:
                logger.info("✅ Answer is clean (no unwanted characters)")
        
        # Overall assessment
        logger.info(f"\n📈 OVERALL ASSESSMENT:")
        logger.info(f"✅ Total questions processed: {len(answers)}")
        logger.info(f"✅ Response time: {elapsed:.2f} seconds")
        
        if elapsed < 10:
            logger.info("✅ Speed target achieved (< 10 seconds)")
        else:
            logger.warning("⚠️  Speed target not achieved (> 10 seconds)")
        
        # Check for unanswered questions
        unanswered_count = sum(1 for answer in answers if not answer or answer.strip() == "")
        if unanswered_count == 0:
            logger.info("✅ All questions were answered")
        else:
            logger.warning(f"⚠️  {unanswered_count} questions were not answered")
        
    else:
        logger.error(f"❌ Request failed: {response.text}")

if __name__ == "__main__":
    asyncio.run(test_malayalam_accuracy())
