def test_always_passes():
    """Basic sanity test"""
    assert 1 + 1 == 2
    
def test_python_version():
    """Test Python version"""
    import sys
    assert sys.version_info >= (3, 9)
