"""
Code parsing functions using Tree-sitter (ADK tools)
All functions return JSON strings for ADK compatibility
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser, Node
from loguru import logger

from src.models import ParsedFile, ParsedFunction, ParsedClass

# Initialize parsers
PY_LANGUAGE = Language(tspython.language())
JS_LANGUAGE = Language(tsjavascript.language())

py_parser = Parser(PY_LANGUAGE)
js_parser = Parser(JS_LANGUAGE)

PARSERS = {
    'python': py_parser,
    'javascript': js_parser,
    'typescript': js_parser
}

def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension"""
    ext_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript'
    }
    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext)

def parse_python_file(file_path: str, content: str) -> str:
    """
    Parse a Python file and extract functions and classes.
    
    Args:
        file_path: Path to the file
        content: File content as string
        
    Returns:
        JSON string with parsed entities (functions, classes, imports)
    """
    try:
        tree = py_parser.parse(bytes(content, 'utf8'))
        root = tree.root_node
        
        # Extract imports
        imports = _extract_imports_python(root, content)
        
        # Extract functions
        functions = _extract_functions_python(root, content, file_path)
        
        # Extract classes
        classes = _extract_classes_python(root, content, file_path)
        
        parsed_file = ParsedFile(
            path=file_path,
            language='python',
            imports=imports,
            functions=functions,
            classes=classes
        )
        
        return json.dumps(parsed_file.model_dump(), default=str)
        
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return json.dumps({"error": str(e), "file_path": file_path})

def _extract_imports_python(root: Node, content: str) -> List[str]:
    """Extract import statements from Python AST"""
    imports = []
    
    def traverse(node):
        if node.type in ['import_statement', 'import_from_statement']:
            import_text = content[node.start_byte:node.end_byte]
            imports.append(import_text)
        for child in node.children:
            traverse(child)
    
    traverse(root)
    return imports

def _extract_functions_python(root: Node, content: str, file_path: str) -> List[ParsedFunction]:
    """Extract function definitions from Python AST"""
    functions = []
    
    def traverse(node):
        if node.type == 'function_definition':
            func = _parse_function_node_python(node, content, file_path)
            if func:
                functions.append(func)
        for child in node.children:
            traverse(child)
    
    traverse(root)
    return functions

def _parse_function_node_python(node: Node, content: str, file_path: str) -> Optional[ParsedFunction]:
    """Parse a single function node"""
    try:
        # Get function name
        name_node = node.child_by_field_name('name')
        name = content[name_node.start_byte:name_node.end_byte] if name_node else "unknown"
        
        # Get parameters
        params_node = node.child_by_field_name('parameters')
        parameters = []
        if params_node:
            for child in params_node.named_children:
                if child.type in ['identifier', 'typed_parameter']:
                    param_name = content[child.start_byte:child.end_byte]
                    parameters.append({'name': param_name.split(':')[0].strip()})
        
        # Get docstring
        docstring = _extract_docstring_python(node, content)
        
        # Get code
        code = content[node.start_byte:node.end_byte]
        
        # Extract function calls
        calls = _extract_function_calls_python(node, content)
        
        # Check if async
        is_async = 'async' in code.split('\n')[0]
        
        return ParsedFunction(
            name=name,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            code=code,
            docstring=docstring,
            parameters=parameters,
            calls=calls,
            is_async=is_async
        )
    except Exception as e:
        logger.warning(f"Error parsing function: {e}")
        return None

def _extract_docstring_python(node: Node, content: str) -> Optional[str]:
    """Extract docstring from function or class"""
    body = node.child_by_field_name('body')
    if not body or not body.named_children:
        return None
    
    first_stmt = body.named_children[0]
    if first_stmt.type == 'expression_statement':
        string_node = first_stmt.named_children[0] if first_stmt.named_children else None
        if string_node and string_node.type == 'string':
            docstring = content[string_node.start_byte:string_node.end_byte]
            return docstring.strip('"""').strip("'''").strip()
    
    return None

def _extract_function_calls_python(node: Node, content: str) -> List[str]:
    """Extract function calls within a node"""
    calls = []
    
    def traverse(n):
        if n.type == 'call':
            func_node = n.child_by_field_name('function')
            if func_node:
                func_name = content[func_node.start_byte:func_node.end_byte]
                # Extract just the function name
                if '.' in func_name:
                    func_name = func_name.split('.')[-1]
                calls.append(func_name)
        for child in n.children:
            traverse(child)
    
    traverse(node)
    return list(set(calls))

def _extract_classes_python(root: Node, content: str, file_path: str) -> List[ParsedClass]:
    """Extract class definitions from Python AST"""
    classes = []
    
    def traverse(node):
        if node.type == 'class_definition':
            cls = _parse_class_node_python(node, content, file_path)
            if cls:
                classes.append(cls)
        for child in node.children:
            traverse(child)
    
    traverse(root)
    return classes

def _parse_class_node_python(node: Node, content: str, file_path: str) -> Optional[ParsedClass]:
    """Parse a single class node"""
    try:
        # Get class name
        name_node = node.child_by_field_name('name')
        name = content[name_node.start_byte:name_node.end_byte] if name_node else "unknown"
        
        # Get base classes
        bases_node = node.child_by_field_name('superclasses')
        base_classes = []
        if bases_node:
            for child in bases_node.named_children:
                base_name = content[child.start_byte:child.end_byte]
                base_classes.append(base_name)
        
        # Get docstring
        docstring = _extract_docstring_python(node, content)
        
        # Get code
        code = content[node.start_byte:node.end_byte]
        
        # Extract methods
        methods = []
        body = node.child_by_field_name('body')
        if body:
            for child in body.named_children:
                if child.type == 'function_definition':
                    method_name_node = child.child_by_field_name('name')
                    if method_name_node:
                        method_name = content[method_name_node.start_byte:method_name_node.end_byte]
                        methods.append(method_name)
        
        return ParsedClass(
            name=name,
            file_path=file_path,
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            code=code,
            docstring=docstring,
            methods=methods,
            base_classes=base_classes
        )
    except Exception as e:
        logger.warning(f"Error parsing class: {e}")
        return None

def generate_node_id(file_path: str, name: str, line: int) -> str:
    """
    Generate unique ID for a code entity.
    
    Args:
        file_path: File path
        name: Entity name
        line: Line number
        
    Returns:
        Unique hash ID
    """
    unique_str = f"{file_path}:{name}:{line}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]

def batch_parse_files(file_paths_and_contents: List[Dict[str, str]]) -> str:
    """
    Parse multiple files in batch.
    
    Args:
        file_paths_and_contents: List of dicts with 'path' and 'content' keys
        
    Returns:
        JSON string with all parsed files
    """
    results = []
    
    for item in file_paths_and_contents:
        file_path = item['path']
        content = item['content']
        
        language = detect_language(file_path)
        if language == 'python':
            parsed = parse_python_file(file_path, content)
            results.append(json.loads(parsed))
    
    return json.dumps({
        "total_files": len(file_paths_and_contents),
        "parsed_files": len(results),
        "files": results
    })