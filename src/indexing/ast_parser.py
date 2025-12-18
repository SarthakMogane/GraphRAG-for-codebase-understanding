"""
AST Parser for Multi-Language Code Analysis
Uses Tree-sitter for robust parsing
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tree_sitter
from tree_sitter import Language, Parser
from loguru import logger
from tree_sitter_languages import get_language, get_parser


@dataclass
class CodeEntity:
    """Represents a parsed code entity (function, class, etc.)"""
    name: str
    type: str  # function, class, method, import
    code: str
    docstring: Optional[str]
    start_line: int
    end_line: int
    file_path: str
    language: str
    parent: Optional[str] = None  # For methods inside classes
    
    def __hash__(self):
        return hash(f"{self.file_path}:{self.type}:{self.name}:{self.start_line}")


class ASTParser:
    """Multi-language AST parser using Tree-sitter"""
    
    # Query patterns for different languages
    PYTHON_QUERIES = """
            ;; 1. DEFINITIONS (The "Nouns")
        (function_definition
            name: (identifier) @func_name
            body: (block) @func_body) @function

        (class_definition
            name: (identifier) @class_name
            body: (block) @class_body) @class

        ;; 2. GLOBALS (The "Constants/Configs")
        ;; Catches 'MAX_RETRIES = 5' or 'app = FastAPI()'
        (assignment
            left: (identifier) @var_name
            right: (_) @var_value) @global_variable

        ;; 3. DEPENDENCIES (The "Verbs" / Edges)
        ;; Catches 'calculate_tax()' inside other code
        (call
            function: [
                (identifier) @call_name
                (attribute attribute: (identifier) @call_name)
            ]) @function_call

        ;; 4. IMPORTS (The "External Links")
        (import_statement) @import
        (import_from_statement) @import_from
    """
    
    JAVASCRIPT_QUERIES = """
            ;; 1. DEFINITIONS (The "Nouns")
            (function_definition
                name: (identifier) @func_name
                body: (block) @func_body) @function

            (class_definition
                name: (identifier) @class_name
                body: (block) @class_body) @class

            ;; 2. GLOBALS (The "Constants/Configs")
            ;; Catches 'MAX_RETRIES = 5' or 'app = FastAPI()'
            (assignment
                left: (identifier) @var_name
                right: (_) @var_value) @global_variable

            ;; 3. DEPENDENCIES (The "Verbs" / Edges)
            ;; Catches 'calculate_tax()' inside other code
            (call
                function: [
                    (identifier) @call_name
                    (attribute attribute: (identifier) @call_name)
                ]) @function_call

            ;; 4. IMPORTS (The "External Links")
            (import_statement) @import
            (import_from_statement) @import_from
    """
    
    def __init__(self, repo_path: str, languages: List[str] = None):
        """
        Initialize parser for a repository
        
        Args:
            repo_path: Path to code repository
            languages: List of languages to parse (default: ['python'])
        """
        self.repo_path = Path(repo_path)
        self.languages = languages or ['python']
        self.parsers = {}
        self.queries = {}
        
        self._setup_parsers()
    
    def _setup_parsers(self):
        """Initialize parsers using ONLY tree_sitter_languages to prevent conflicts"""
        try:
            # We use the unified wrapper for everything.
            # This avoids the "1 argument vs 2 argument" crash.
            from tree_sitter_languages import get_language, get_parser

            for lang in self.languages:
                # 1. Get the Parser (Engine)
                # get_parser() automatically returns a ready-to-use parser object
                self.parsers[lang] = get_parser(lang)
                
                # 2. Get the Language (Grammar)
                # We need this object to compile the queries
                lang_obj = get_language(lang)
                
                # 3. Load the correct query string
                if lang == 'python':
                    query_str = self.PYTHON_QUERIES
                elif lang == 'javascript':
                    query_str = self.JAVASCRIPT_QUERIES
                else:
                    query_str = None

                # 4. Compile the query (if string exists)
                if query_str:
                    try:
                        self.queries[lang] = lang_obj.query(query_str)
                    except Exception as q_err:
                        logger.error(f"Query compilation failed for {lang}: {q_err}")

            logger.info(f"Initialized parsers for: {list(self.parsers.keys())}")

        except Exception as e:
            logger.error(f"Tree-sitter setup failed: {e}")
            logger.info("Falling back to Python AST module")
            self._use_fallback_parser()
    def _use_fallback_parser(self):
        """Fallback to Python's built-in ast module"""
        self.parsers['python'] = 'ast'  # Marker for fallback
    
    def parse_repository(self) -> List[CodeEntity]:
        """
        Parse entire repository and extract all entities
        
        Returns:
            List of CodeEntity objects
        """
        entities = []
        
        for lang in self.languages:
            ext = self._get_extension(lang)
            files = self._find_files(ext)
            
            logger.info(f"Found {len(files)} {lang} files")
            
            for file_path in files:
                try:
                    file_entities = self.parse_file(file_path, lang)
                    entities.extend(file_entities)
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {e}")
        
        logger.info(f"Extracted {len(entities)} code entities")
        return entities
    
    def parse_file(self, file_path: Path, language: str) -> List[CodeEntity]:
        """Parse a single file and extract entities"""
        
        with open(file_path, 'rb') as f:
            code = f.read()
        
        # Use appropriate parser
        if self.parsers.get(language) == 'ast':
            return self._parse_python_ast(file_path, code)
        elif language in self.parsers:
            return self._parse_treesitter(file_path, code, language)
        else:
            logger.warning(f"No parser for {language}")
            return []
    
    def _parse_treesitter(self, file_path: Path, code: bytes, 
                          language: str) -> List[CodeEntity]:
        """Parse using Tree-sitter"""
        entities = []
        parser = self.parsers[language]
        query = self.queries[language]
        
        tree = parser.parse(code)
        captures = query.captures(tree.root_node)
        
        code_str = code.decode('utf-8')
        lines = code_str.split('\n')
        
        for node, capture_name in captures:
            if capture_name in ['function', 'class']:
                entity = self._extract_entity_treesitter(
                    node, capture_name, file_path, lines, language
                )
                if entity:
                    entities.append(entity)
            elif capture_name in ['import', 'import_from']:
                entity = self._extract_import_treesitter(
                    node, file_path, lines, language
                )
                if entity:
                    entities.append(entity)
        
        return entities
    
    def _parse_python_ast(self, file_path: Path, code: bytes) -> List[CodeEntity]:
        """Fallback parser using Python's ast module"""
        import ast
        
        entities = []
        code_str = code.decode('utf-8')
        
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return []
        
        lines = code_str.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                entity = CodeEntity(
                    name=node.name,
                    type='function',
                    code=self._get_code_slice(lines, node.lineno, node.end_lineno),
                    docstring=ast.get_docstring(node),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    file_path=str(file_path),
                    language='python'
                )
                entities.append(entity)
            
            elif isinstance(node, ast.ClassDef):
                entity = CodeEntity(
                    name=node.name,
                    type='class',
                    code=self._get_code_slice(lines, node.lineno, node.end_lineno),
                    docstring=ast.get_docstring(node),
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    file_path=str(file_path),
                    language='python'
                )
                entities.append(entity)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_name = self._extract_import_name(node)
                entity = CodeEntity(
                    name=import_name,
                    type='import',
                    code=ast.get_source_segment(code_str, node) or '',
                    docstring=None,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    file_path=str(file_path),
                    language='python'
                )
                entities.append(entity)
        
        return entities
    
    def _extract_entity_treesitter(self, node, entity_type: str, 
                                    file_path: Path, lines: List[str],
                                    language: str) -> Optional[CodeEntity]:
        """Extract entity details from Tree-sitter node"""
        
        # Get name (child with 'name' field)
        name_node = node.child_by_field_name('name')
        print("name_node:",name_node) ## that give row and column not name  for ex. <Node type="identifier", start_point=(4, 4), end_point=(4, 15)>
        if not name_node:
            return None
        
        name = self._get_node_text(name_node, lines)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        
        # Get code
        code = self._get_code_slice(lines, start_line, end_line)
        
        # Try to extract docstring
        docstring = self._extract_docstring(node, lines, language)
        
        return CodeEntity(
            name=name,
            type=entity_type,
            code=code,
            docstring=docstring,
            start_line=start_line,
            end_line=end_line,
            file_path=str(file_path),
            language=language
        )
    
    def _extract_import_treesitter(self, node, file_path: Path, 
                                    lines: List[str], language: str) -> Optional[CodeEntity]:
        """Extract import statement"""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        code = self._get_code_slice(lines, start_line, end_line)
        
        # Extract module name from code
        import_name = code.split()[1] if len(code.split()) > 1 else 'unknown'
        
        return CodeEntity(
            name=import_name,
            type='import',
            code=code,
            docstring=None,
            start_line=start_line,
            end_line=end_line,
            file_path=str(file_path),
            language=language
        )
    
    def _extract_docstring(self, node, lines: List[str], language: str) -> Optional[str]:
        """Extract docstring from function/class"""
        if language == 'python':
            body = node.child_by_field_name('body')
            if body and body.child_count > 0:
                first_child = body.child(0)
                if first_child.type == 'expression_statement':
                    expr = first_child.child(0)
                    if expr and expr.type == 'string':
                        return self._get_node_text(expr, lines).strip('"\'')
        return None
    
    def _get_node_text(self, node, lines: List[str]) -> str:
        """Get text content of a Tree-sitter node"""
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        start_col = node.start_point[1]
        end_col = node.end_point[1]
        
        if start_line == end_line:
            return lines[start_line][start_col:end_col]
        else:
            result = lines[start_line][start_col:]
            for i in range(start_line + 1, end_line):
                result += '\n' + lines[i]
            result += '\n' + lines[end_line][:end_col]
            return result
    
    def _get_code_slice(self, lines: List[str], start: int, end: int) -> str:
        """Get code between line numbers"""
        return '\n'.join(lines[start-1:end])
    
    def _extract_import_name(self, node) -> str:
        """Extract module name from import node"""
        import ast
        if isinstance(node, ast.Import):
            return node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            return node.module or 'unknown'
        return 'unknown'
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java',
            'cpp': '.cpp'
        }
        return extensions.get(language, '.py')
    
    def _find_files(self, extension: str) -> List[Path]:
        """Find all files with given extension"""
        ignore_patterns = [
            '__pycache__', 'node_modules', '.git', 
            'venv', 'env', '.pytest_cache', 'test', 'tests'
        ]
        
        files = []
        for path in self.repo_path.rglob(f'*{extension}'):
            # Check if path contains ignored patterns
            if not any(pattern in str(path) for pattern in ignore_patterns):
                files.append(path)
        
        return files


# Example usage
if __name__ == "__main__":
    import sys
    
    # if len(sys.argv) < 2:
    #     print("Usage: python ast_parser.py <repo_path>")
    #     sys.exit(1)
    

    # repo_path = sys.argv[1]
    repo_path = 'd:/kaggle_project/GraphRAG/data/repositories/SMS-Spam-VotingClassifier-'
    parser = ASTParser(repo_path)
    entities = parser.parse_repository()
    
    print(f"\n=== Parsing Results ===")
    print(f"Total entities: {len(entities)}")
    
    # Group by type
    from collections import Counter
    type_counts = Counter(e.type for e in entities)
    for entity_type, count in type_counts.items():
        print(f"{entity_type}: {count}")
    
    # Show sample
    print(f"\n=== Sample Entities ===")
    for entity in entities[:5]:
        print(f"\n{entity.type.upper()}: {entity.name}")
        print(f"  File: {entity.file_path}")
        print(f"  Lines: {entity.start_line}-{entity.end_line}")
        if entity.docstring:
            print(f"  Doc: {entity.docstring[:80]}...")