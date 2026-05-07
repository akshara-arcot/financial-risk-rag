import pytest
from rag_pipeline import chunk_documents, generate_embeddings

def test_chunk_documents():
    """Test document chunking"""
    sample_text = "This is a test document. " * 100
    chunks = chunk_documents(sample_text, chunk_size=512)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 512 for chunk in chunks)

def test_generate_embeddings():
    """Test embedding generation"""
    chunks = ["Test chunk 1", "Test chunk 2"]
    embeddings = generate_embeddings(chunks)
    
    assert len(embeddings) == len(chunks)
    assert len(embeddings[0]) > 0  # Has dimensions
