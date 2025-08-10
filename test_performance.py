#!/usr/bin/env python3
"""
Performance test script for the optimized RAG pipeline
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

async def test_performance():
    """Test the performance of the optimized RAG pipeline"""
    
    # Test client
    client = TestClient(app)
    
    # Test data
    test_request = QueryRequest(
        documents=["https://example.com/test-document.pdf"],
        questions=[
            "What is the policy coverage?",
            "How much is the premium?",
            "What are the waiting periods?",
            "എന്താണ് പോളിസി കവറേജ്?",  # Malayalam question
            "പ്രീമിയം എത്രയാണ്?"  # Malayalam question
        ]
    )
    
    logger.info("🚀 Starting performance test...")
    
    # Test multiple runs to check caching
    for run in range(3):
        logger.info(f"\n📊 Test Run {run + 1}")
        
        start_time = time.time()
        
        # Make request
        response = client.post("/hackrx/run", json=test_request.dict())
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        logger.info(f"⏱️  Response time: {elapsed:.2f} seconds")
        logger.info(f"📝 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            answers = data.get('answers', [])
            logger.info(f"✅ Generated {len(answers)} answers")
            
            # Check for unwanted characters
            for i, answer in enumerate(answers):
                if any(char in answer for char in ['🔸', '🔹', '⚡', '✨', '💡']):
                    logger.warning(f"⚠️  Answer {i+1} contains unwanted characters")
                else:
                    logger.info(f"✅ Answer {i+1} is clean")
        else:
            logger.error(f"❌ Request failed: {response.text}")
        
        # Wait between runs
        if run < 2:
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(test_performance())
