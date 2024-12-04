import pytest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_streamlit():
    """Mock streamlit for testing"""
    import streamlit as st
    # Create mock session state
    if not hasattr(st, 'session_state'):
        setattr(st, 'session_state', {})
    return st

@pytest.fixture
def test_config():
    """Test configuration values"""
    return {
        'test_pdf_path': 'tests/test_files/test.pdf',
        'test_docx_path': 'tests/test_files/test.docx',
        'test_image_path': 'tests/test_files/test.png',
        'test_model': 'gpt-3.5-turbo'
    }
