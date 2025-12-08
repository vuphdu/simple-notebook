"""
Tests for the chunking module.
"""
import pytest
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import DocumentChunker, DocumentChunk


class TestDocumentChunker:
    """Tests for DocumentChunker class."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = DocumentChunker()
        text = "This is a test document. " * 50
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = DocumentChunker()
        
        chunks = chunker.chunk_text("")
        
        assert chunks == []
    
    def test_chunk_text_with_metadata(self):
        """Test chunking with metadata."""
        chunker = DocumentChunker()
        text = "Test content " * 100
        metadata = {"author": "test", "category": "testing"}
        
        chunks = chunker.chunk_text(text, source_file="test.txt", metadata=metadata)
        
        assert len(chunks) > 0
        assert chunks[0].metadata.get("author") == "test"
        assert chunks[0].source_file == "test.txt"
    
    def test_chunk_file_txt(self):
        """Test chunking a text file."""
        chunker = DocumentChunker()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content " * 100)
            temp_path = Path(f.name)
        
        try:
            chunks = chunker.chunk_file(temp_path)
            
            assert len(chunks) > 0
            assert chunks[0].metadata.get("file_extension") == ".txt"
        finally:
            temp_path.unlink()
    
    def test_chunk_file_not_found(self):
        """Test chunking non-existent file."""
        chunker = DocumentChunker()
        
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file(Path("/nonexistent/file.txt"))


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""
    
    def test_chunk_creation(self):
        """Test creating a DocumentChunk."""
        chunk = DocumentChunk(
            chunk_id="test_0",
            content="Test content",
            metadata={"key": "value"},
            start_index=0,
            end_index=12
        )
        
        assert chunk.chunk_id == "test_0"
        assert chunk.content == "Test content"
        assert chunk.metadata["key"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
