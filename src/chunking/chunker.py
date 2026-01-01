"""
Document Chunking Module

This module provides functionality to split documents into smaller chunks
for efficient vectorization and retrieval.
"""
from typing import Optional
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, ChunkingConfig


class DocumentChunk(BaseModel):
    """Represents a single chunk of a document."""
    chunk_id: str
    content: str
    metadata: dict
    start_index: int
    end_index: int
    source_file: Optional[str] = None
    # OlmOCR-style classification fields
    chunk_type: str = "text"       # text, code, table, heading, list
    has_equations: bool = False    # Contains LaTeX or math expressions
    has_code_blocks: bool = False  # Contains code blocks or code-like content


class DocumentChunker:
    """
    Handles document chunking with configurable parameters.
    
    Supports multiple chunking strategies:
    - Character-based chunking
    - Token-based chunking
    - Semantic chunking (paragraph/sentence aware)
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the DocumentChunker.
        
        Args:
            config: Chunking configuration. Uses default settings if not provided.
        """
        self.config = config or settings.chunking
        self._text_splitter = self._create_text_splitter()
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create the text splitter based on configuration."""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
            is_separator_regex=False,
        )
    
    def _detect_chunk_type(self, content: str) -> dict:
        """
        Detect chunk type and characteristics for OlmOCR-style classification.
        
        Args:
            content: The chunk content to analyze.
            
        Returns:
            dict with chunk_type, has_equations, has_code_blocks
        """
        result = {
            "chunk_type": "text",
            "has_equations": False,
            "has_code_blocks": False,
        }
        
        content_stripped = content.strip()
        
        # Detect LaTeX equations
        latex_markers = [r"\(", r"\)", r"\[", r"\]", "$$", r"\begin{", r"\end{"]
        if any(marker in content for marker in latex_markers):
            result["has_equations"] = True
        
        # Detect code blocks (markdown style)
        if "```" in content:
            result["has_code_blocks"] = True
            result["chunk_type"] = "code"
        
        # Detect code-like content (function definitions, etc.)
        code_patterns = [
            "def ", "function ", "class ", "import ", "#include",
            "public ", "private ", "void ", "int ", "return ",
            "if (", "for (", "while (", "=>", "->",
        ]
        if any(pattern in content for pattern in code_patterns):
            result["has_code_blocks"] = True
            if result["chunk_type"] == "text":
                result["chunk_type"] = "code"
        
        # Detect tables (markdown or ascii)
        lines = content.split('\n')
        pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        if pipe_lines >= 2:
            result["chunk_type"] = "table"
        
        # Detect headings (markdown style)
        if content_stripped.startswith('#'):
            result["chunk_type"] = "heading"
        
        # Detect lists
        list_markers = ['- ', '* ', '+ ', '1. ', '2. ', '3. ']
        list_lines = sum(1 for line in lines if any(line.strip().startswith(m) for m in list_markers))
        if list_lines >= 3 and list_lines / max(len(lines), 1) > 0.5:
            result["chunk_type"] = "list"
        
        return result
    
    def chunk_text(
        self,
        text: str,
        source_file: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> list[DocumentChunk]:
        """
        Split text into chunks with OlmOCR-style classification.
        
        Args:
            text: The text content to chunk.
            source_file: Optional source file path for metadata.
            metadata: Additional metadata to attach to chunks.
            
        Returns:
            List of DocumentChunk objects with type classification.
        """
        if not text.strip():
            return []
        
        # Use langchain's text splitter
        chunks = self._text_splitter.split_text(text)
        
        result = []
        current_index = 0
        
        for i, chunk_content in enumerate(chunks):
            # Find the actual position in the original text
            start_idx = text.find(chunk_content, current_index)
            if start_idx == -1:
                start_idx = current_index
            end_idx = start_idx + len(chunk_content)
            
            # Detect chunk type (OlmOCR enhancement)
            chunk_classification = self._detect_chunk_type(chunk_content)
            
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_type": chunk_classification["chunk_type"],
                "has_equations": chunk_classification["has_equations"],
                "has_code_blocks": chunk_classification["has_code_blocks"],
                **(metadata or {})
            }
            
            chunk = DocumentChunk(
                chunk_id=f"{source_file or 'unknown'}_{i}",
                content=chunk_content,
                metadata=chunk_metadata,
                start_index=start_idx,
                end_index=end_idx,
                source_file=source_file,
                # OlmOCR fields
                chunk_type=chunk_classification["chunk_type"],
                has_equations=chunk_classification["has_equations"],
                has_code_blocks=chunk_classification["has_code_blocks"],
            )
            result.append(chunk)
            current_index = start_idx + 1
        
        return result
    
    def chunk_file(self, file_path: Path) -> list[DocumentChunk]:
        """
        Read and chunk a file.
        
        Args:
            file_path: Path to the file to chunk.
            
        Returns:
            List of DocumentChunk objects.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content based on extension
        content = self._read_file(file_path)
        
        metadata = {
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size": file_path.stat().st_size,
        }
        
        return self.chunk_text(
            text=content,
            source_file=str(file_path),
            metadata=metadata
        )
    
    def _read_file(self, file_path: Path) -> str:
        """
        Read file content based on file type.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            File content as string.
        """
        suffix = file_path.suffix.lower()
        
        # Plain text files (including markdown)
        if suffix in ['.txt', '.md', '.markdown']:
            return file_path.read_text(encoding='utf-8')
        
        # Source code files - read as plain text with file info header
        CODE_EXTENSIONS = [
            '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx',  # C/C++
            '.py', '.pyi',  # Python
            '.java',  # Java
            '.js', '.jsx', '.ts', '.tsx', '.mjs',  # JavaScript/TypeScript
            '.go',  # Go
            '.rs',  # Rust
            '.rb',  # Ruby
            '.php',  # PHP
            '.swift', '.kt', '.kts',  # Swift/Kotlin
            '.cs',  # C#
            '.scala',  # Scala
            '.lua',  # Lua
            '.sh', '.bash', '.zsh', '.fish',  # Shell
            '.sql',  # SQL
            '.r', '.R',  # R
            '.m', '.mm',  # Objective-C
            '.pl', '.pm',  # Perl
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',  # Config
            '.xml', '.html', '.htm', '.css', '.scss', '.sass',  # Web
            '.cmake', '.make', '.mk', 'Makefile',  # Build
        ]
        
        if suffix in CODE_EXTENSIONS:
            content = file_path.read_text(encoding='utf-8')
            # Add file info header for better context
            header = f"// File: {file_path.name}\n// Path: {file_path}\n\n"
            return header + content
        
        elif suffix == '.pdf':
            return self._read_pdf(file_path)
        
        elif suffix == '.docx':
            return self._read_docx(file_path)
        
        else:
            # Try to read as text
            try:
                return file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file type: {suffix}")
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file content."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF processing")
    
    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file content."""
        try:
            from docx import Document
            doc = Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise ImportError("python-docx is required for DOCX processing")
    
    def chunk_directory(
        self,
        directory: Path,
        extensions: Optional[list[str]] = None,
        recursive: bool = True
    ) -> list[DocumentChunk]:
        """
        Chunk all documents in a directory.
        
        Args:
            directory: Path to the directory.
            extensions: List of file extensions to process. Default: all supported.
            recursive: Whether to process subdirectories.
            
        Returns:
            List of all DocumentChunk objects from the directory.
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        extensions = extensions or ['.txt', '.md', '.pdf', '.docx', '.markdown']
        pattern = '**/*' if recursive else '*'
        
        all_chunks = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    chunks = self.chunk_file(file_path)
                    all_chunks.extend(chunks)
                    print(f"Chunked: {file_path.name} -> {len(chunks)} chunks")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return all_chunks


# Convenience function
def chunk_documents(
    source: str | Path,
    config: Optional[ChunkingConfig] = None
) -> list[DocumentChunk]:
    """
    Chunk documents from a file or directory.
    
    Args:
        source: Path to file or directory.
        config: Optional chunking configuration.
        
    Returns:
        List of DocumentChunk objects.
    """
    chunker = DocumentChunker(config)
    source_path = Path(source)
    
    if source_path.is_file():
        return chunker.chunk_file(source_path)
    elif source_path.is_dir():
        return chunker.chunk_directory(source_path)
    else:
        raise ValueError(f"Invalid source path: {source}")
