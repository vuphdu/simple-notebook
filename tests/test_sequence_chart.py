"""
Tests for the sequence chart module.
"""
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sequence_chart import SequenceChartProcessor, SequenceChart


class TestSequenceChartProcessor:
    """Tests for SequenceChartProcessor class."""
    
    def test_parse_mermaid_basic(self):
        """Test parsing a basic Mermaid sequence diagram."""
        processor = SequenceChartProcessor()
        
        content = """sequenceDiagram
    participant A
    participant B
    A->>B: Hello
    B-->>A: Hi there"""
        
        chart = processor.parse_mermaid(content)
        
        assert isinstance(chart, SequenceChart)
        assert "A" in chart.participants
        assert "B" in chart.participants
        assert len(chart.elements) == 2
    
    def test_parse_mermaid_with_title(self):
        """Test parsing diagram with title."""
        processor = SequenceChartProcessor()
        
        content = """sequenceDiagram
    title Test Diagram
    participant User
    User->>Server: Request"""
        
        chart = processor.parse_mermaid(content)
        
        assert chart.title == "Test Diagram"
    
    def test_extract_from_markdown(self):
        """Test extracting charts from markdown."""
        processor = SequenceChartProcessor()
        
        markdown = """
# Document

Some text here.

```mermaid
sequenceDiagram
    A->>B: Message
```

More text.

```mermaid
sequenceDiagram
    C->>D: Another message
```
"""
        
        charts = processor.extract_from_markdown(markdown)
        
        assert len(charts) == 2
    
    def test_generate_description(self):
        """Test generating chart description."""
        processor = SequenceChartProcessor()
        
        content = """sequenceDiagram
    participant Client
    participant Server
    Client->>Server: GET /api/data
    Server-->>Client: JSON response"""
        
        chart = processor.parse_mermaid(content)
        description = chart.generate_description()
        
        assert "Client" in description
        assert "Server" in description
        assert "GET /api/data" in description
    
    def test_get_vectorization_data(self):
        """Test preparing data for vectorization."""
        processor = SequenceChartProcessor()
        
        content = """sequenceDiagram
    A->>B: Test message"""
        
        charts = [processor.parse_mermaid(content)]
        data = processor.get_vectorization_data(charts)
        
        assert len(data) == 1
        assert "content" in data[0]
        assert "metadata" in data[0]
        assert data[0]["metadata"]["type"] == "sequence_chart"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
