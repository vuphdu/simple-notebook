"""
Sequence Chart Processing Module

This module provides functionality to parse, render, and vectorize
sequence diagrams (e.g., Mermaid sequence charts).
"""
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field
import re
import subprocess
import tempfile
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import settings, SequenceChartConfig, DOCUMENTS_DIR, ensure_directories


@dataclass
class SequenceChartElement:
    """Represents an element in a sequence chart."""
    element_type: str  # participant, message, note, etc.
    source: Optional[str] = None
    target: Optional[str] = None
    message: Optional[str] = None
    description: Optional[str] = None


@dataclass
class SequenceChart:
    """Represents a parsed sequence chart."""
    chart_id: str
    title: Optional[str] = None
    participants: list[str] = field(default_factory=list)
    elements: list[SequenceChartElement] = field(default_factory=list)
    raw_content: str = ""
    description: str = ""
    image_path: Optional[str] = None
    source_file: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "participants": self.participants,
            "description": self.description,
            "image_path": self.image_path,
            "source_file": self.source_file,
            "element_count": len(self.elements)
        }
    
    def generate_description(self) -> str:
        """Generate a textual description of the sequence chart."""
        parts = []
        
        if self.title:
            parts.append(f"Sequence Diagram: {self.title}")
        
        if self.participants:
            parts.append(f"Participants: {', '.join(self.participants)}")
        
        # Describe the flow
        flow_description = []
        for elem in self.elements:
            if elem.element_type == "message" and elem.source and elem.target:
                msg = elem.message or "communicates"
                flow_description.append(f"{elem.source} sends '{msg}' to {elem.target}")
            elif elem.element_type == "note" and elem.description:
                flow_description.append(f"Note: {elem.description}")
        
        if flow_description:
            parts.append("Flow:\n" + "\n".join(f"  - {fd}" for fd in flow_description))
        
        self.description = "\n\n".join(parts)
        return self.description


class SequenceChartProcessor:
    """
    Handles sequence chart parsing, rendering, and vectorization.
    
    Supports Mermaid sequence diagram syntax.
    """
    
    def __init__(self, config: Optional[SequenceChartConfig] = None):
        """
        Initialize the SequenceChartProcessor.
        
        Args:
            config: Sequence chart configuration.
        """
        self.config = config or settings.sequence_chart
        ensure_directories()
    
    def parse_mermaid(self, content: str, source_file: Optional[str] = None) -> SequenceChart:
        """
        Parse a Mermaid sequence diagram.
        
        Args:
            content: Mermaid sequence diagram content.
            source_file: Optional source file path.
            
        Returns:
            Parsed SequenceChart object.
        """
        # Generate chart ID
        chart_id = hashlib.md5(content.encode()).hexdigest()[:8]
        
        chart = SequenceChart(
            chart_id=chart_id,
            raw_content=content,
            source_file=source_file
        )
        
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and diagram declaration
            if not line or line.startswith('sequenceDiagram'):
                continue
            
            # Parse title
            if line.startswith('title'):
                chart.title = line.replace('title', '').strip()
                continue
            
            # Parse participant
            participant_match = re.match(r'participant\s+(\w+)(?:\s+as\s+(.+))?', line)
            if participant_match:
                participant_name = participant_match.group(2) or participant_match.group(1)
                chart.participants.append(participant_name)
                continue
            
            # Parse message arrows (various arrow types)
            message_patterns = [
                r'(\w+)\s*->>>\s*(\w+)\s*:\s*(.+)',  # ->>>
                r'(\w+)\s*-->>\s*(\w+)\s*:\s*(.+)',  # -->>
                r'(\w+)\s*->>?\s*(\w+)\s*:\s*(.+)',  # ->> or ->
                r'(\w+)\s*-->\s*(\w+)\s*:\s*(.+)',   # -->
            ]
            
            for pattern in message_patterns:
                message_match = re.match(pattern, line)
                if message_match:
                    elem = SequenceChartElement(
                        element_type="message",
                        source=message_match.group(1),
                        target=message_match.group(2),
                        message=message_match.group(3)
                    )
                    chart.elements.append(elem)
                    
                    # Auto-add participants if not already added
                    for p in [elem.source, elem.target]:
                        if p and p not in chart.participants:
                            chart.participants.append(p)
                    break
            
            # Parse notes
            note_match = re.match(r'Note\s+(left|right|over)\s+(?:of\s+)?(\w+)\s*:\s*(.+)', line, re.IGNORECASE)
            if note_match:
                elem = SequenceChartElement(
                    element_type="note",
                    target=note_match.group(2),
                    description=note_match.group(3)
                )
                chart.elements.append(elem)
        
        # Generate description
        chart.generate_description()
        
        return chart
    
    def extract_from_markdown(self, markdown_content: str, source_file: Optional[str] = None) -> list[SequenceChart]:
        """
        Extract sequence charts from markdown content.
        
        Args:
            markdown_content: Markdown content with mermaid code blocks.
            source_file: Optional source file path.
            
        Returns:
            List of parsed SequenceChart objects.
        """
        charts = []
        
        # Find mermaid code blocks
        pattern = r'```mermaid\s*\n(.*?)\n```'
        matches = re.findall(pattern, markdown_content, re.DOTALL)
        
        for match in matches:
            if 'sequenceDiagram' in match:
                chart = self.parse_mermaid(match, source_file)
                charts.append(chart)
        
        return charts
    
    def export_image(
        self,
        chart: SequenceChart,
        output_dir: Optional[Path] = None,
        format: Optional[str] = None
    ) -> Optional[str]:
        """
        Export sequence chart to image.
        
        Uses mermaid-cli (mmdc) if available, otherwise creates a placeholder.
        
        Args:
            chart: The SequenceChart to export.
            output_dir: Output directory for the image.
            format: Image format (png, svg, pdf).
            
        Returns:
            Path to the exported image, or None if export failed.
        """
        output_dir = output_dir or DOCUMENTS_DIR / "charts"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        format = format or self.config.export_format
        output_file = output_dir / f"chart_{chart.chart_id}.{format}"
        
        # Try using mermaid-cli
        try:
            result = self._export_with_mmdc(chart, output_file)
            if result:
                chart.image_path = str(output_file)
                return str(output_file)
        except Exception as e:
            print(f"mermaid-cli export failed: {e}")
        
        # Fallback: create a text-based placeholder
        try:
            result = self._create_placeholder_image(chart, output_file)
            if result:
                chart.image_path = str(output_file)
                return str(output_file)
        except Exception as e:
            print(f"Placeholder image creation failed: {e}")
        
        return None
    
    def _export_with_mmdc(self, chart: SequenceChart, output_file: Path) -> bool:
        """Export using mermaid-cli (mmdc)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(chart.raw_content)
            temp_input = f.name
        
        try:
            cmd = ['mmdc', '-i', temp_input, '-o', str(output_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        finally:
            Path(temp_input).unlink(missing_ok=True)
    
    def _create_placeholder_image(self, chart: SequenceChart, output_file: Path) -> bool:
        """Create a simple placeholder image with chart info."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create image
            width, height = 800, 600
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw title
            y_offset = 20
            if chart.title:
                draw.text((20, y_offset), f"Sequence: {chart.title}", fill='black')
                y_offset += 30
            
            # Draw participants
            draw.text((20, y_offset), f"Participants: {', '.join(chart.participants)}", fill='gray')
            y_offset += 40
            
            # Draw elements summary
            for i, elem in enumerate(chart.elements[:10]):
                if elem.element_type == "message":
                    text = f"{elem.source} → {elem.target}: {elem.message}"
                else:
                    text = f"[{elem.element_type}] {elem.description or ''}"
                draw.text((40, y_offset + i * 25), text[:60], fill='black')
            
            # Save
            img.save(str(output_file))
            return True
            
        except ImportError:
            print("PIL not available for placeholder image creation")
            return False
    
    def process_file(
        self,
        file_path: Path,
        export_images: bool = True
    ) -> list[SequenceChart]:
        """
        Process a file to extract and optionally export sequence charts.
        
        Args:
            file_path: Path to the file to process.
            export_images: Whether to export images.
            
        Returns:
            List of processed SequenceChart objects.
        """
        file_path = Path(file_path)
        content = file_path.read_text(encoding='utf-8')
        
        charts = self.extract_from_markdown(content, str(file_path))
        
        if export_images:
            for chart in charts:
                self.export_image(chart)
        
        print(f"Processed {file_path.name}: found {len(charts)} sequence charts")
        return charts
    
    def get_vectorization_data(self, charts: list[SequenceChart]) -> list[dict]:
        """
        Prepare chart data for vectorization.
        
        The description and metadata will be used for vector embedding.
        
        Args:
            charts: List of SequenceChart objects.
            
        Returns:
            List of dicts ready for vectorization.
        """
        data = []
        for chart in charts:
            data.append({
                "chunk_id": f"chart_{chart.chart_id}",
                "content": chart.description,
                "metadata": {
                    "type": "sequence_chart",
                    "title": chart.title or "",
                    "participants": ", ".join(chart.participants),
                    "image_path": chart.image_path or "",
                    "source_file": chart.source_file or ""
                },
                "source_file": chart.source_file
            })
        return data


# Global processor instance
_processor: Optional[SequenceChartProcessor] = None


def get_chart_processor() -> SequenceChartProcessor:
    """Get or create the global chart processor instance."""
    global _processor
    if _processor is None:
        _processor = SequenceChartProcessor()
    return _processor
