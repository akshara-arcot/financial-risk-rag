import pytest
from retriever import search_similar_chunks

def test_search_similar_chunks():
    """Test retrieval returns results"""
    query = "What is the revenue?"
    results = search_similar_chunks(query, top_k=3)
    
    assert isinstance(results, list)
    assert len(results) <= 3

def test_empty_query():
    """Test handling of empty queries"""
    results = search_similar_chunks("", top_k=3)
    
    assert results == [] or len(results) == 0
