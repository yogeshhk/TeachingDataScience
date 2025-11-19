"""
Lightweight PDF parser using Docling with minimal dependencies.
Falls back to simpler extraction if layout models aren't available.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ParsedTable:
    """Represents a parsed table with structure preserved."""
    table_name: str
    page_number: int
    rows: int
    columns: int
    data: List[List[str]]  # 2D array of cell values
    headers: List[str]
    markdown: str
    caption: Optional[str] = None
    section: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "table_name": self.table_name,
            "page_number": self.page_number,
            "rows": self.rows,
            "columns": self.columns,
            "data": self.data,
            "headers": self.headers,
            "markdown": self.markdown,
            "caption": self.caption,
            "section": self.section
        }


@dataclass
class ParsedText:
    """Represents a parsed text section."""
    content: str
    page_number: int
    section: Optional[str] = None
    subsection: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "page_number": self.page_number,
            "section": self.section,
            "subsection": self.subsection
        }


@dataclass
class ParsedImage:
    """Represents a parsed image/chart."""
    page_number: int
    image_type: str  # 'chart', 'logo', 'diagram', etc.
    ocr_text: Optional[str] = None
    caption: Optional[str] = None
    section: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "page_number": self.page_number,
            "image_type": self.image_type,
            "ocr_text": self.ocr_text,
            "caption": self.caption,
            "section": self.section,
            "width": self.width,
            "height": self.height
        }


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""
    filename: str
    total_pages: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[ParsedTable] = field(default_factory=list)
    text_sections: List[ParsedText] = field(default_factory=list)
    images: List[ParsedImage] = field(default_factory=list)
    raw_text: str = ""
    parse_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "filename": self.filename,
            "total_pages": self.total_pages,
            "metadata": self.metadata,
            "tables": [t.to_dict() for t in self.tables],
            "text_sections": [t.to_dict() for t in self.text_sections],
            "images": [i.to_dict() for i in self.images],
            "raw_text": self.raw_text,
            "parse_timestamp": self.parse_timestamp.isoformat()
        }


class SimpleDoclingParser:
    """
    Lightweight PDF parser using Docling without heavy layout models.
    """
    
    def __init__(self):
        """Initialize the simple parser."""
        logger.info("SimpleDoclingParser initialized (lightweight mode)")
    
    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file with basic text extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument containing extracted content
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Parsing PDF (simple mode): {pdf_file.name}")
        
        try:
            # Try Docling first with simple options
            from docling.document_converter import DocumentConverter
            
            # Create converter with minimal options (no heavy models)
            converter = DocumentConverter()
            
            # Convert PDF
            result = converter.convert(pdf_path)
            doc = result.document
            
            # Initialize parsed document
            parsed_doc = ParsedDocument(
                filename=pdf_file.name,
                total_pages=len(doc.pages) if hasattr(doc, 'pages') else 1
            )
            
            # Get markdown export (simple text extraction)
            parsed_doc.raw_text = doc.export_to_markdown()
            
            # Basic text extraction
            parsed_doc.text_sections = [
                ParsedText(
                    content=parsed_doc.raw_text,
                    page_number=1,
                    section="Full Document"
                )
            ]
            
            logger.info(f"Successfully parsed {pdf_file.name} (simple mode)")
            logger.info(f"Extracted {len(parsed_doc.raw_text)} characters")
            
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Error with Docling: {e}")
            # Fallback to PyMuPDF
            return self._fallback_parse(pdf_path)
    
    def _fallback_parse(self, pdf_path: str) -> ParsedDocument:
        """
        Fallback parser using PyMuPDF for basic text extraction.
        """
        logger.info("Using fallback parser (PyMuPDF)")
        
        try:
            import fitz  # PyMuPDF
            
            pdf_file = Path(pdf_path)
            doc = fitz.open(pdf_path)
            
            parsed_doc = ParsedDocument(
                filename=pdf_file.name,
                total_pages=len(doc)
            )
            
            all_text = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                all_text.append(text)
                
                parsed_doc.text_sections.append(
                    ParsedText(
                        content=text,
                        page_number=page_num,
                        section=f"Page {page_num}"
                    )
                )
            
            parsed_doc.raw_text = "\n\n".join(all_text)
            
            logger.info(f"Successfully parsed {pdf_file.name} with fallback")
            logger.info(f"Extracted {len(parsed_doc.raw_text)} characters from {len(doc)} pages")
            
            doc.close()
            return parsed_doc
            
        except ImportError:
            raise ImportError("Neither Docling nor PyMuPDF is available for PDF parsing")
    
    def save_parsed_output(self, parsed_doc: ParsedDocument, output_dir: str) -> None:
        """
        Save parsed document to JSON file.
        
        Args:
            parsed_doc: ParsedDocument object
            output_dir: Directory to save output
        """
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on original PDF name
        output_file = output_path / f"{Path(parsed_doc.filename).stem}_parsed.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_doc.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved parsed output to {output_file}")


if __name__ == "__main__":
    # Quick test
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils.logger import setup_logger
    
    setup_logger("simple_parser", Path("logs/simple_parser.log"), logging.INFO)
    
    parser = SimpleDoclingParser()
    pdf_path = "bajaj_finserv_factsheet_Oct.pdf"
    
    if Path(pdf_path).exists():
        parsed_doc = parser.parse_pdf(pdf_path)
        print(f"\n✓ Parsed: {parsed_doc.filename}")
        print(f"  Pages: {parsed_doc.total_pages}")
        print(f"  Text length: {len(parsed_doc.raw_text)} chars")
        print(f"  Text sections: {len(parsed_doc.text_sections)}")
        
        # Save
        parser.save_parsed_output(parsed_doc, "data/processed")
        print(f"✓ Saved to data/processed/")
    else:
        print(f"✗ PDF not found: {pdf_path}")
