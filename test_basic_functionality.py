#!/usr/bin/env python3
"""
Basic functionality test for the RAG pipeline
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

def test_basic_functionality():
    """Test basic functionality without requiring API key"""
    
    client = TestClient(app)
    
    # Test health endpoint
    logger.info("🔍 Testing health endpoint...")
    health_response = client.get("/health")
    logger.info(f"Health status: {health_response.status_code}")
    
    # Test root endpoint
    logger.info("🔍 Testing root endpoint...")
    root_response = client.get("/")
    logger.info(f"Root status: {root_response.status_code}")
    
    # Test cache stats endpoint
    logger.info("🔍 Testing cache stats endpoint...")
    cache_response = client.get("/cache/stats")
    logger.info(f"Cache stats status: {cache_response.status_code}")
    
    # Test metrics endpoint
    logger.info("🔍 Testing metrics endpoint...")
    metrics_response = client.get("/metrics")
    logger.info(f"Metrics status: {metrics_response.status_code}")
    
    # Test the main endpoint with a simple request
    logger.info("🔍 Testing main endpoint...")
    test_request = QueryRequest(
        documents="https://example.com/test-document.pdf",
        questions=["What is this document about?"]
    )
    
    try:
        response = client.post("/api/v1/hackrx/run", json=test_request.model_dump())
        logger.info(f"Main endpoint status: {response.status_code}")
        if response.status_code != 200:
            logger.info(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Error testing main endpoint: {e}")
    
    logger.info("✅ Basic functionality test completed")

if __name__ == "__main__":
    test_basic_functionality()
