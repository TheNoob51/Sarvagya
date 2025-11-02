"""
Pytest configuration and shared fixtures for integration tests.
"""

import pytest
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    # Set test environment variables to avoid real API calls
    os.environ['GEMINI_API_KEY'] = 'test_key'
    os.environ['SERPAPI_KEY'] = 'test_key'
    os.environ['DEBUG'] = 'false'
    
    yield
    
    # Cleanup after tests
    pass