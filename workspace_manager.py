import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Union
import glob

class WorkspaceManager:
    DEFAULT_WORKSPACE = "Workspace"
    
    def __init__(self, root_dir: str):
        """Initialize workspace manager with the root directory."""
        self.root_dir = Path(root_dir)
        self.workspace_dir = self.root_dir / self.DEFAULT_WORKSPACE
        self.config_dir = self.workspace_dir / '.aiworkstation'
        self._ensure_workspace_exists()
        
    def _ensure_workspace_exists(self):
        """Ensure the workspace and configuration directories exist."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def add_file_to_workspace(self, file_path: Union[str, Path], create_subdirs: bool = True) -> bool:
        """
        Add a file to the workspace, optionally preserving directory structure.
        Returns True if successful, False otherwise.
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                logging.error(f"Source file does not exist: {file_path}")
                return False
                
            if create_subdirs:
                # Try to maintain relative path structure
                if source_path.is_absolute():
                    # For absolute paths, just use the file name
                    target_path = self.workspace_dir / source_path.name
                else:
                    # For relative paths, maintain the structure
                    target_path = self.workspace_dir / source_path
            else:
                # Just copy to workspace root
                target_path = self.workspace_dir / source_path.name
                
            # Create parent directories if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_path, target_path)
            logging.info(f"Added file to workspace: {target_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding file to workspace: {str(e)}")
            return False
            
    def remove_from_workspace(self, file_path: Union[str, Path]) -> bool:
        """Remove a file from the workspace."""
        try:
            target_path = self.workspace_dir / Path(file_path).name
            if target_path.exists():
                target_path.unlink()
                logging.info(f"Removed file from workspace: {target_path}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error removing file from workspace: {str(e)}")
            return False
            
    def list_workspace_files(self, pattern: str = "*") -> List[Path]:
        """List all files in the workspace, optionally filtered by pattern."""
        try:
            return [p for p in self.workspace_dir.rglob(pattern) 
                   if p.is_file() and '.aiworkstation' not in str(p)]
        except Exception as e:
            logging.error(f"Error listing workspace files: {str(e)}")
            return []
            
    def get_workspace_path(self) -> Path:
        """Get the path to the workspace directory."""
        return self.workspace_dir
        
    def save_workspace_config(self, config: Dict) -> bool:
        """Save workspace-specific configuration."""
        try:
            config_file = self.config_dir / 'workspace_config.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving workspace config: {str(e)}")
            return False
            
    def load_workspace_config(self) -> Dict:
        """Load workspace-specific configuration."""
        config_file = self.config_dir / 'workspace_config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading workspace config: {str(e)}")
        return {}
        
    def get_project_files(self, file_patterns: List[str] = None) -> List[str]:
        """Get list of project files, optionally filtered by patterns."""
        if not file_patterns:
            file_patterns = ['**/*.py', '**/*.js', '**/*.java', '**/*.cpp', '**/*.h']
            
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(str(self.workspace_dir / pattern), recursive=True))
        return [f for f in files if '.aiworkstation' not in f]
        
    def analyze_codebase(self) -> Dict:
        """Analyze the codebase to gather detailed context about the project.
        
        Returns:
            Dict containing:
            - file_types: Distribution of file types
            - total_files: Total number of files
            - languages: Programming languages used
            - dependencies: External dependencies
            - complexity_metrics: Code complexity indicators
            - documentation_coverage: Documentation status
            - test_coverage: Test file distribution
        """
        analysis = {
            'file_types': {},
            'total_files': 0,
            'languages': set(),
            'dependencies': set(),
            'complexity_metrics': {
                'avg_file_size': 0,
                'max_file_size': 0,
                'largest_files': []
            },
            'documentation_coverage': {
                'files_with_docs': 0,
                'total_docstrings': 0
            },
            'test_coverage': {
                'test_files': 0,
                'test_directories': set()
            }
        }
        
        total_size = 0
        file_sizes = []
        
        for file in self.get_project_files():
            path = Path(file)
            ext = path.suffix
            size = path.stat().st_size
            
            # Basic file statistics
            analysis['file_types'][ext] = analysis['file_types'].get(ext, 0) + 1
            analysis['total_files'] += 1
            total_size += size
            file_sizes.append((size, file))
            
            # Track largest files
            if len(analysis['complexity_metrics']['largest_files']) < 5:
                analysis['complexity_metrics']['largest_files'].append((file, size))
            else:
                min_size = min(f[1] for f in analysis['complexity_metrics']['largest_files'])
                if size > min_size:
                    analysis['complexity_metrics']['largest_files'] = sorted(
                        [f for f in analysis['complexity_metrics']['largest_files'] if f[1] != min_size] + [(file, size)],
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
            
            # Test coverage analysis
            if 'test' in path.stem.lower() or 'test' in str(path.parent).lower():
                analysis['test_coverage']['test_files'] += 1
                analysis['test_coverage']['test_directories'].add(str(path.parent))
            
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    
                    # Language-specific analysis
                    if ext == '.py':
                        analysis['languages'].add('Python')
                        # Extract Python imports and analyze complexity
                        import_count = 0
                        class_count = 0
                        func_count = 0
                        doc_count = 0
                        
                        lines = content.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith('import ') or line.startswith('from '):
                                import_count += 1
                                pkg = line.split()[1].split('.')[0]
                                analysis['dependencies'].add(pkg)
                            elif line.startswith('class '):
                                class_count += 1
                            elif line.startswith('def '):
                                func_count += 1
                            elif line.strip().startswith('"""') or line.strip().startswith("'''"):
                                doc_count += 1
                        
                        if doc_count > 0:
                            analysis['documentation_coverage']['files_with_docs'] += 1
                            analysis['documentation_coverage']['total_docstrings'] += doc_count
                        
                    elif ext in ['.js', '.ts']:
                        analysis['languages'].add('JavaScript' if ext == '.js' else 'TypeScript')
                        # JavaScript/TypeScript specific analysis
                        if 'import ' in content or 'require(' in content:
                            for line in content.split('\n'):
                                if 'import ' in line or 'require(' in line:
                                    analysis['dependencies'].add(line.strip())
                        
                    elif ext in ['.java']:
                        analysis['languages'].add('Java')
                        # Java specific analysis
                        if 'import ' in content:
                            for line in content.split('\n'):
                                if line.strip().startswith('import '):
                                    analysis['dependencies'].add(line.strip())
                    
                    elif ext in ['.cpp', '.hpp', '.h']:
                        analysis['languages'].add('C++')
                        # C++ specific analysis
                        if '#include ' in content:
                            for line in content.split('\n'):
                                if line.strip().startswith('#include '):
                                    analysis['dependencies'].add(line.strip())
                
            except Exception as e:
                logging.warning(f"Error analyzing {file}: {str(e)}")
        
        # Calculate complexity metrics
        if analysis['total_files'] > 0:
            analysis['complexity_metrics']['avg_file_size'] = total_size / analysis['total_files']
            analysis['complexity_metrics']['max_file_size'] = max(file_sizes)[0]
        
        # Convert sets to lists for JSON serialization
        analysis['languages'] = list(analysis['languages'])
        analysis['dependencies'] = list(analysis['dependencies'])
        analysis['test_coverage']['test_directories'] = list(analysis['test_coverage']['test_directories'])
        
        return analysis

    def get_file_context(self, file_path: str, context_lines: int = 5) -> Dict:
        """Get detailed context information for a specific file."""
        try:
            file_path = Path(file_path)
            if not file_path.is_absolute():
                file_path = self.workspace_dir / file_path
                
            context = {
                'file_name': file_path.name,
                'file_type': file_path.suffix,
                'size': file_path.stat().st_size,
                'last_modified': file_path.stat().st_mtime,
                'related_files': [],
                'imports': [],
                'symbols': [],
                'complexity_metrics': {
                    'lines_of_code': 0,
                    'comment_lines': 0,
                    'blank_lines': 0,
                    'function_count': 0,
                    'class_count': 0,
                    'import_count': 0
                },
                'documentation': {
                    'has_docstrings': False,
                    'docstring_count': 0,
                    'todo_count': 0
                }
            }
            
            # Find related files with similar names
            parent = file_path.parent
            base_name = file_path.stem
            for f in parent.glob(f"{base_name}*"):
                if f != file_path:
                    context['related_files'].append(str(f.relative_to(self.workspace_dir)))
            
            # Analyze file content
            with open(file_path, 'r') as f:
                content = f.readlines()
                
            in_multiline_comment = False
            
            for line in content:
                line = line.strip()
                
                # Count basic metrics
                if line:
                    context['complexity_metrics']['lines_of_code'] += 1
                else:
                    context['complexity_metrics']['blank_lines'] += 1
                
                # Track comments and documentation
                if line.startswith('#') or line.startswith('//'):
                    context['complexity_metrics']['comment_lines'] += 1
                    if 'TODO' in line:
                        context['documentation']['todo_count'] += 1
                elif '"""' in line or "'''" in line:
                    context['documentation']['has_docstrings'] = True
                    context['documentation']['docstring_count'] += 1
                    in_multiline_comment = not in_multiline_comment
                elif in_multiline_comment:
                    context['complexity_metrics']['comment_lines'] += 1
                
                # Track code structure
                if line.startswith('def '):
                    context['complexity_metrics']['function_count'] += 1
                elif line.startswith('class '):
                    context['complexity_metrics']['class_count'] += 1
                elif line.startswith('import ') or line.startswith('from '):
                    context['complexity_metrics']['import_count'] += 1
                    context['imports'].append(line.strip())
            
            return context
        except Exception as e:
            logging.error(f"Error getting file context: {str(e)}")
            return {}

    def suggest_code_improvements(self, file_path: str) -> List[Dict]:
        """Generate detailed code improvement suggestions for a file."""
        try:
            context = self.get_file_context(file_path)
            suggestions = []
            
            # Analyze code structure
            if context['complexity_metrics']['lines_of_code'] > 500:
                suggestions.append({
                    'type': 'structure',
                    'message': 'File is quite large. Consider splitting into smaller modules',
                    'severity': 'warning'
                })
            
            if context['complexity_metrics']['import_count'] > 15:
                suggestions.append({
                    'type': 'dependency',
                    'message': 'High number of imports. Consider refactoring to reduce dependencies',
                    'severity': 'warning'
                })
            
            # Documentation analysis
            if not context['documentation']['has_docstrings']:
                suggestions.append({
                    'type': 'documentation',
                    'message': 'File lacks docstrings. Consider adding documentation',
                    'severity': 'suggestion'
                })
            
            if context['documentation']['todo_count'] > 0:
                suggestions.append({
                    'type': 'maintenance',
                    'message': f"Found {context['documentation']['todo_count']} TODO comments that need attention",
                    'severity': 'info'
                })
            
            # Code quality checks
            if context['complexity_metrics']['function_count'] > 0:
                avg_loc_per_func = context['complexity_metrics']['lines_of_code'] / context['complexity_metrics']['function_count']
                if avg_loc_per_func > 50:
                    suggestions.append({
                        'type': 'complexity',
                        'message': 'Functions are quite long. Consider breaking them into smaller functions',
                        'severity': 'suggestion'
                    })
            
            # Comment density check
            comment_ratio = context['complexity_metrics']['comment_lines'] / max(context['complexity_metrics']['lines_of_code'], 1)
            if comment_ratio < 0.1:
                suggestions.append({
                    'type': 'documentation',
                    'message': 'Low comment density. Consider adding more inline documentation',
                    'severity': 'suggestion'
                })
            
            return suggestions
        except Exception as e:
            logging.error(f"Error analyzing file for improvements: {str(e)}")
            return []
