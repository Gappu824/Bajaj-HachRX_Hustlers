#!/usr/bin/env python3
"""
Quick performance test for speed validation
"""

import time
import logging
from app.main import app
from app.models.query import QueryRequest
from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_performance():
    """Test performance with a simple request"""
    
    client = TestClient(app)
    
    # Simple test request
    test_request = QueryRequest(
        documents="https://hackrx.blob.core.windows.net/hackrx/rounds/News.pdf?sv=2023-01-03&spr=https&st=2025-08-07T17%3A10%3A11Z&se=2026-08-08T17%3A10%3A00Z&sr=b&sp=r&sig=ybRsnfv%2B6VbxPz5xF7kLLjC4ehU0NF7KDkXua9ujSf0%3D",
        questions=[
            "What was Apple's investment commitment?",
            "ട്രംപ് ഏത് ദിവസമാണ് 100% ശുൽകം പ്രഖ്യാപിച്ചത്?"
        ]
    )
    
    logger.info("🚀 Starting performance test...")
    
    start_time = time.time()
    
    try:
        response = client.post("/api/v1/hackrx/run", json=test_request.model_dump())
        end_time = time.time()
        elapsed = end_time - start_time
        
        logger.info(f"⏱️  Response time: {elapsed:.2f} seconds")
        logger.info(f"📝 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get('answers', [])
            logger.info(f"✅ Generated {len(answers)} answers")
            
            # Check if answers are in correct language
            if len(answers) >= 2:
                # Check English answer
                english_answer = answers[0]
                malayalam_chars = len([c for c in english_answer if '\u0d00' <= c <= '\u0d7f'])
                if malayalam_chars == 0:
                    logger.info("✅ English answer is in English")
                else:
                    logger.warning("⚠️  English answer contains Malayalam")
                
                # Check Malayalam answer
                malayalam_answer = answers[1]
                malayalam_chars = len([c for c in malayalam_answer if '\u0d00' <= c <= '\u0d7f'])
                if malayalam_chars > len(malayalam_answer) * 0.1:
                    logger.info("✅ Malayalam answer is in Malayalam")
                else:
                    logger.warning("⚠️  Malayalam answer is not in Malayalam")
            
            # Performance assessment
            if elapsed < 10:
                logger.info("✅ Speed target achieved (< 10 seconds)")
            else:
                logger.warning(f"⚠️  Speed target not achieved ({elapsed:.2f} seconds)")
                
        else:
            logger.error(f"❌ Request failed: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_performance()
