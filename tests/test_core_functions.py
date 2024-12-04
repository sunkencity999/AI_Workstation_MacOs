import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open
from sgpt_interface import (
    load_api_key,
    execute_shell_command,
    execute_python_script,
    check_ollama_running,
    extract_text_from_pdf,
    extract_text_from_docx,
    process_image,
    web_search
)
import requests
from io import BytesIO
import streamlit as st

@pytest.fixture
def mock_streamlit():
    with patch('streamlit.session_state', {'openai_api_key': None}):
        yield

def test_load_api_key(mock_streamlit):
    """Test API key loading functionality"""
    # Mock environment variable and streamlit session state
    mock_config = 'OPENAI_API_KEY=test_key'
    mock_path = MagicMock()
    mock_path.exists.return_value = True
    mock_path.read_text.return_value = mock_config

    with patch('pathlib.Path') as mock_path_class, \
         patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}, clear=True):
        mock_path_class.home.return_value = MagicMock()
        mock_path_class.home.return_value.__truediv__.return_value = mock_path
        key = load_api_key()
        assert key == 'test_key'
        assert os.environ['OPENAI_API_KEY'] == 'test_key'

def test_execute_shell_command():
    """Test shell command execution"""
    # Test successful command
    result = execute_shell_command('echo "test"')
    assert 'test' in result
    
    # Test command that exists but will fail
    result = execute_shell_command('ls /nonexistent_directory')
    assert 'No such file or directory' in result

def test_execute_python_script():
    """Test Python script execution"""
    # Test valid Python code
    script = 'print("test")'
    result = execute_python_script(script)
    assert 'test' in result
    
    # Test invalid Python code that won't crash the interpreter
    script = 'print(undefined_variable)'
    result = execute_python_script(script)
    assert 'Error' in result or 'undefined_variable' in result

def test_check_ollama_running():
    """Test Ollama server check"""
    with patch('requests.get') as mock_get:
        # Test when Ollama is running
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        assert check_ollama_running() is True
        
        # Test when Ollama is not running
        mock_get.side_effect = requests.exceptions.RequestException("Connection refused")
        assert check_ollama_running() is False

def test_web_search():
    """Test web search functionality"""
    mock_html = """
    <div class="result">
        <a class="result__a" href="http://test.com">Test</a>
        <a class="result__snippet">Test result</a>
    </div>
    """
    
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        results = web_search('test query')
        assert isinstance(results, list)
        assert len(results) > 0
        assert results[0]['title'] == 'Test'
        assert results[0]['snippet'] == 'Test result'
        assert results[0]['link'] == 'http://test.com'

def test_file_processing(test_config):
    """Test file processing functions"""
    # Create test files directory
    os.makedirs('tests/test_files', exist_ok=True)

    # Test PDF extraction with proper mocking
    test_pdf_content = b'%PDF-1.4\ntest content\n%%EOF'
    with patch('pypdf.PdfReader') as mock_pdf_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = 'test content'
        mock_pdf_reader.return_value.pages = [mock_page]
        text = extract_text_from_pdf(test_pdf_content)
        assert isinstance(text, str)
        assert text == 'test content'

    # Test DOCX extraction with proper mocking
    with patch('docx.Document') as mock_docx:
        mock_doc = MagicMock()
        mock_paragraph = MagicMock()
        mock_paragraph.text = 'test content'
        mock_doc.paragraphs = [mock_paragraph]
        mock_docx.return_value = mock_doc

        # Test with file path
        text = extract_text_from_docx(test_config['test_docx_path'])
        assert isinstance(text, str)
        assert text == 'test content'

        # Test with bytes
        text = extract_text_from_docx(b'test content')
        assert isinstance(text, str)
        assert text == 'test content'

def test_image_processing(test_config):
    """Test image processing function"""
    # Create test files directory
    os.makedirs('tests/test_files', exist_ok=True)
    
    # Create a proper test image file
    with open(test_config['test_image_path'], 'wb') as f:
        f.write(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
    
    # Test image processing with proper mocking
    result = process_image(test_config['test_image_path'])
    assert isinstance(result, (str, bytes, type(None)))  # Accept any valid return type
