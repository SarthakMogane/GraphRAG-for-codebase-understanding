"""
GitHub repository operations (ADK tools)
All functions return JSON strings for ADK compatibility
"""
import json
from pathlib import Path
from typing import List
from git import Repo
from loguru import logger
import shutil

from src.config.settings import settings

def clone_repository(repo_url: str, branch: str = "main") -> str:
    """
    Clone a GitHub repository.
    
    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/user/repo)
        branch: Branch to clone (default: main)
        
    Returns:
        JSON string with clone status and local path
    """
    try:
        # Extract repo name from URL
        repo_name = repo_url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        local_path = settings.REPO_CACHE_DIR / repo_name
        
        # Check if already exists
        if local_path.exists():
            logger.info(f"Repository already exists: {local_path}")
            return json.dumps({
                "status": "exists",
                "repo_name": repo_name,
                "local_path": str(local_path),
                "message": "Repository already cloned"
            })
        
        # Clone repository
        logger.info(f"Cloning {repo_url} to {local_path}")
        Repo.clone_from(
            repo_url,
            local_path,
            branch=branch,
            depth=1  # Shallow clone for speed
        )
        
        return json.dumps({
            "status": "success",
            "repo_name": repo_name,
            "local_path": str(local_path),
            "branch": branch
        })
        
    except Exception as e:
        logger.error(f"Clone failed: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)
        })

def get_source_files(repo_path: str, extensions: List[str] = None) -> str:
    """
    Get all source files from a repository.
    
    Args:
        repo_path: Path to repository
        extensions: File extensions to include (default: ['.py', '.js', '.ts'])
        
    Returns:
        JSON string with list of file paths
    """
    if extensions is None:
        extensions = ['.py', '.js', '.jsx', '.ts', '.tsx']
    
    try:
        repo_path = Path(repo_path)
        
        # Directories to exclude
        exclude_dirs = {
            '.git', '__pycache__', 'node_modules', 'venv', 
            'env', 'build', 'dist', '.pytest_cache', 'coverage',
            '.next', 'out', 'target'
        }
        
        source_files = []
        
        for ext in extensions:
            for file_path in repo_path.rglob(f'*{ext}'):
                # Check if file is in excluded directory
                if any(excluded in file_path.parts for excluded in exclude_dirs):
                    continue
                
                source_files.append(str(file_path))
        
        return json.dumps({
            "total_files": len(source_files),
            "files": source_files,
            "extensions": extensions
        })
        
    except Exception as e:
        logger.error(f"Error getting source files: {e}")
        return json.dumps({
            "error": str(e),
            "total_files": 0,
            "files": []
        })

def read_file_content(file_path: str) -> str:
    """
    Read content of a source file.
    
    Args:
        file_path: Path to file
        
    Returns:
        JSON string with file content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return json.dumps({
            "file_path": file_path,
            "content": content,
            "size": len(content),
            "lines": len(content.split('\n'))
        })
        
    except UnicodeDecodeError:
        logger.warning(f"Failed to decode {file_path}")
        return json.dumps({
            "error": "decode_error",
            "file_path": file_path
        })
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return json.dumps({
            "error": str(e),
            "file_path": file_path
        })

def get_repository_stats(repo_path: str) -> str:
    """
    Get statistics about a repository.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        JSON string with repository statistics
    """
    try:
        repo_path = Path(repo_path)
        
        # Get all source files
        files_result = json.loads(get_source_files(str(repo_path)))
        source_files = files_result.get('files', [])
        
        stats = {
            'repo_path': str(repo_path),
            'total_files': len(source_files),
            'by_extension': {},
            'total_lines': 0,
            'total_size_bytes': 0
        }
        
        for file_path in source_files:
            ext = Path(file_path).suffix
            stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1
            
            try:
                content_result = json.loads(read_file_content(file_path))
                if 'content' in content_result:
                    stats['total_lines'] += content_result['lines']
                    stats['total_size_bytes'] += content_result['size']
            except:
                pass
        
        return json.dumps(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return json.dumps({"error": str(e)})

def delete_repository(repo_path: str) -> str:
    """
    Delete a cloned repository.
    
    Args:
        repo_path: Path to repository
        
    Returns:
        JSON string with deletion status
    """
    try:
        repo_path = Path(repo_path)
        
        if repo_path.exists():
            shutil.rmtree(repo_path)
            return json.dumps({
                "status": "success",
                "message": f"Deleted {repo_path}"
            })
        else:
            return json.dumps({
                "status": "not_found",
                "message": f"Repository not found: {repo_path}"
            })
            
    except Exception as e:
        logger.error(f"Error deleting repository: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e)
        })