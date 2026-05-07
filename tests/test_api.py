import pytest
from fastapi.testclient import TestClient
from api import app  # or from app import app, depending on your setup

client = TestClient(app)

def test_health_check():
    """Test health endpoint exists"""
    response = client.get("/health")
    assert response.status_code == 200

def test_query_endpoint():
    """Test main query endpoint"""
    response = client.post("/query", json={"query": "test", "top_k": 3})
    
    # Should return 200 or 422, not 500
    assert response.status_code in [200, 422]
