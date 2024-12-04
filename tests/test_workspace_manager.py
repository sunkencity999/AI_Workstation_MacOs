import pytest
import os
import tempfile
from pathlib import Path
from workspace_manager import WorkspaceManager

@pytest.fixture
def temp_root():
    """Create a temporary root directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def workspace_manager(temp_root):
    """Create a workspace manager instance with a temporary root directory."""
    manager = WorkspaceManager(temp_root)
    return manager

def create_test_file(path: Path, content: str = "test content"):
    """Helper function to create a test file."""
    path.write_text(content)
    return path

def test_workspace_initialization(workspace_manager, temp_root):
    """Test workspace manager initialization and default workspace creation."""
    workspace_path = Path(temp_root) / "Workspace"
    assert workspace_path.exists()
    assert workspace_path.is_dir()
    assert (workspace_path / '.aiworkstation').exists()

def test_add_file_to_workspace(workspace_manager, temp_root):
    """Test adding a file to the workspace."""
    # Create a test file
    test_file = Path(temp_root) / "test.txt"
    create_test_file(test_file)
    
    # Add file to workspace
    assert workspace_manager.add_file_to_workspace(test_file)
    
    # Verify file was copied to workspace
    workspace_file = workspace_manager.get_workspace_path() / "test.txt"
    assert workspace_file.exists()
    assert workspace_file.read_text() == "test content"

def test_add_file_with_subdirs(workspace_manager, temp_root):
    """Test adding a file while preserving directory structure."""
    # Create a test file in a subdirectory
    subdir = Path(temp_root) / "subdir"
    subdir.mkdir()
    test_file = subdir / "test.txt"
    create_test_file(test_file)
    
    # Add file to workspace with subdirs
    assert workspace_manager.add_file_to_workspace(test_file, create_subdirs=True)
    
    # Verify file and directory structure
    workspace_file = workspace_manager.get_workspace_path() / "subdir" / "test.txt"
    assert workspace_file.exists()
    assert workspace_file.read_text() == "test content"

def test_remove_from_workspace(workspace_manager, temp_root):
    """Test removing a file from the workspace."""
    # Add a file first
    test_file = Path(temp_root) / "test.txt"
    create_test_file(test_file)
    workspace_manager.add_file_to_workspace(test_file)
    
    # Remove the file
    assert workspace_manager.remove_from_workspace("test.txt")
    
    # Verify file was removed
    workspace_file = workspace_manager.get_workspace_path() / "test.txt"
    assert not workspace_file.exists()

def test_list_workspace_files(workspace_manager, temp_root):
    """Test listing workspace files."""
    # Add multiple files
    files = ["test1.txt", "test2.py", "test3.js"]
    for filename in files:
        test_file = Path(temp_root) / filename
        create_test_file(test_file)
        workspace_manager.add_file_to_workspace(test_file)
    
    # List all files
    workspace_files = workspace_manager.list_workspace_files()
    assert len(workspace_files) == 3
    
    # List with pattern
    py_files = workspace_manager.list_workspace_files("*.py")
    assert len(py_files) == 1
    assert py_files[0].name == "test2.py"

def test_get_project_files(workspace_manager, temp_root):
    """Test getting project files."""
    # Add multiple files
    files = ["test1.txt", "test2.py", "test3.js"]
    for filename in files:
        test_file = Path(temp_root) / filename
        create_test_file(test_file)
        workspace_manager.add_file_to_workspace(test_file)
    
    # Get project files
    project_files = workspace_manager.get_project_files()
    assert len(project_files) == 3
    
def test_analyze_codebase(workspace_manager, temp_root):
    """Test codebase analysis."""
    # Add multiple files
    files = ["test1.txt", "test2.py", "test3.js"]
    for filename in files:
        test_file = Path(temp_root) / filename
        create_test_file(test_file)
        workspace_manager.add_file_to_workspace(test_file)
    
    # Analyze codebase
    analysis = workspace_manager.analyze_codebase()
    assert analysis['total_files'] == 3
    assert '.py' in analysis['file_types']
    assert '.js' in analysis['file_types']
    assert 'Python' in analysis['languages']
    assert 'JavaScript' in analysis['languages']

def test_get_file_context(workspace_manager, temp_root):
    """Test getting file context."""
    # Add multiple files
    files = ["test1.txt", "test2.py", "test3.js"]
    for filename in files:
        test_file = Path(temp_root) / filename
        create_test_file(test_file)
        workspace_manager.add_file_to_workspace(test_file)
    
    # Get file context
    context = workspace_manager.get_file_context('test2.py')
    assert context['file_name'] == 'test2.py'
    assert context['file_type'] == '.py'
    assert len(context['related_files']) == 0

def test_suggest_code_improvements(workspace_manager, temp_root):
    """Test code improvement suggestions."""
    # Add multiple files
    files = ["test1.txt", "test2.py", "test3.js"]
    for filename in files:
        test_file = Path(temp_root) / filename
        create_test_file(test_file)
        workspace_manager.add_file_to_workspace(test_file)
    
    # Suggest code improvements
    suggestions = workspace_manager.suggest_code_improvements(str(workspace_manager.get_workspace_path() / 'test2.py'))
    assert len(suggestions) == 0
