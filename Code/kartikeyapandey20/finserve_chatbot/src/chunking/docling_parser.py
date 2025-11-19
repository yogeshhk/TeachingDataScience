"""
Docling-based PDF parser for extracting structured content from fund factsheets.

This module uses Docling's advanced layout analysis to extract:
- Text paragraphs with semantic boundaries
- Tables with preserved structure (rows/columns)
- Charts and images with OCR
- Document metadata (NAV, AUM, dates)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, TextItem

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


class DoclingParser:
    """
    Advanced PDF parser using Docling for multimodal content extraction.
    
    Features:
    - Layout-aware text extraction
    - Table structure preservation
    - OCR for charts and images
    - Section hierarchy detection
    - Metadata extraction
    """
    
    def __init__(
        self,
        use_ocr: bool = True,
        extract_images: bool = True,
        image_mode: ImageRefMode = ImageRefMode.PLACEHOLDER
    ):
        """
        Initialize the Docling parser.
        
        Args:
            use_ocr: Enable OCR for images and charts
            extract_images: Extract images from PDF
            image_mode: How to handle images (PLACEHOLDER, EMBEDDED, or REFERENCED)
        """
        self.use_ocr = use_ocr
        self.extract_images = extract_images
        self.image_mode = image_mode
        
        # Configure pipeline options for optimal extraction
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = use_ocr
        pipeline_options.do_table_structure = True  # Critical for table extraction
        
        # Initialize DocumentConverter with options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        
        logger.info(
            f"DoclingParser initialized (OCR: {use_ocr}, "
            f"Extract Images: {extract_images})"
        )
    
    def parse_pdf(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF file and extract all structured content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument containing all extracted content
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Parsing PDF: {pdf_file.name}")
        
        # Convert PDF using Docling
        result = self.converter.convert(pdf_path)
        doc = result.document
        
        # Initialize parsed document
        parsed_doc = ParsedDocument(
            filename=pdf_file.name,
            total_pages=len(doc.pages) if hasattr(doc, 'pages') else 1
        )
        
        # Extract metadata
        parsed_doc.metadata = self._extract_metadata(doc)
        
        # Extract tables
        parsed_doc.tables = self._extract_tables(doc)
        logger.info(f"Extracted {len(parsed_doc.tables)} tables")
        
        # Extract text sections
        parsed_doc.text_sections = self._extract_text(doc)
        logger.info(f"Extracted {len(parsed_doc.text_sections)} text sections")
        
        # Extract images/charts
        if self.extract_images:
            parsed_doc.images = self._extract_images(doc)
            logger.info(f"Extracted {len(parsed_doc.images)} images/charts")
        
        # Get full text for reference
        parsed_doc.raw_text = doc.export_to_markdown()
        
        logger.info(f"Successfully parsed {pdf_file.name}")
        return parsed_doc
    
    def _extract_metadata(self, doc) -> Dict[str, Any]:
        """
        Extract document-level metadata.
        
        Args:
            doc: Docling document object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract from document properties if available
        if hasattr(doc, 'metadata'):
            metadata.update(doc.metadata)
        
        # Try to extract fund-specific metadata from first page
        # This will be enhanced by the fund_name_extractor module
        first_page_text = ""
        if hasattr(doc, 'pages') and len(doc.pages) > 0:
            first_page_text = doc.pages[0].export_to_markdown()[:1000]
        
        # Basic metadata extraction (will be enhanced)
        metadata['first_page_preview'] = first_page_text
        metadata['extraction_date'] = datetime.now().isoformat()
        
        return metadata
    
    def _extract_tables(self, doc) -> List[ParsedTable]:
        """
        Extract all tables with preserved structure.
        
        Args:
            doc: Docling document object
            
        Returns:
            List of ParsedTable objects
        """
        tables = []
        
        # Iterate through document elements
        for element, _level in doc.iterate_items():
            if isinstance(element, TableItem):
                try:
                    # Export table to DataFrame for easy manipulation
                    df = element.export_to_dataframe()
                    
                    if df is None or df.empty:
                        logger.warning("Empty table found, skipping")
                        continue
                    
                    # Convert DataFrame to structured format
                    headers = df.columns.tolist()
                    data = df.values.tolist()
                    
                    # Generate markdown representation
                    markdown = element.export_to_markdown()
                    
                    # Get table metadata
                    page_num = element.prov[0].page_no if element.prov else 1
                    caption = element.caption.text if hasattr(element, 'caption') and element.caption else None
                    
                    # Create ParsedTable object
                    parsed_table = ParsedTable(
                        table_name=f"Table_Page{page_num}_{len(tables)+1}",
                        page_number=page_num,
                        rows=len(data),
                        columns=len(headers),
                        data=data,
                        headers=headers,
                        markdown=markdown,
                        caption=caption
                    )
                    
                    tables.append(parsed_table)
                    
                except Exception as e:
                    logger.error(f"Error extracting table: {str(e)}")
                    continue
        
        return tables
    
    def _extract_text(self, doc) -> List[ParsedText]:
        """
        Extract text sections with hierarchy.
        
        Args:
            doc: Docling document object
            
        Returns:
            List of ParsedText objects
        """
        text_sections = []
        current_section = None
        current_text = []
        
        for element, level in doc.iterate_items():
            if isinstance(element, TextItem):
                text_content = element.text.strip()
                
                if not text_content:
                    continue
                
                page_num = element.prov[0].page_no if element.prov else 1
                
                # Detect if this is a heading (based on level or text characteristics)
                is_heading = (
                    level == 0 or 
                    (len(text_content) < 100 and text_content.isupper()) or
                    element.label == "section_header" if hasattr(element, 'label') else False
                )
                
                if is_heading:
                    # Save previous section if exists
                    if current_text:
                        parsed_text = ParsedText(
                            content=" ".join(current_text),
                            page_number=page_num,
                            section=current_section
                        )
                        text_sections.append(parsed_text)
                        current_text = []
                    
                    # Start new section
                    current_section = text_content
                else:
                    # Add to current section
                    current_text.append(text_content)
        
        # Add final section
        if current_text:
            parsed_text = ParsedText(
                content=" ".join(current_text),
                page_number=1,
                section=current_section
            )
            text_sections.append(parsed_text)
        
        return text_sections
    
    def _extract_images(self, doc) -> List[ParsedImage]:
        """
        Extract images and charts with OCR.
        
        Args:
            doc: Docling document object
            
        Returns:
            List of ParsedImage objects
        """
        images = []
        
        for element, _level in doc.iterate_items():
            if isinstance(element, PictureItem):
                try:
                    page_num = element.prov[0].page_no if element.prov else 1
                    
                    # Get caption if available
                    caption = element.caption.text if hasattr(element, 'caption') and element.caption else None
                    
                    # OCR text if available
                    ocr_text = None
                    if self.use_ocr and hasattr(element, 'text'):
                        ocr_text = element.text
                    
                    # Detect image type (basic heuristic)
                    image_type = "unknown"
                    if caption:
                        caption_lower = caption.lower()
                        if any(word in caption_lower for word in ['chart', 'graph', 'plot']):
                            image_type = "chart"
                        elif 'allocation' in caption_lower:
                            image_type = "allocation_chart"
                        elif 'performance' in caption_lower:
                            image_type = "performance_chart"
                    
                    parsed_image = ParsedImage(
                        page_number=page_num,
                        image_type=image_type,
                        ocr_text=ocr_text,
                        caption=caption
                    )
                    
                    images.append(parsed_image)
                    
                except Exception as e:
                    logger.error(f"Error extracting image: {str(e)}")
                    continue
        
        return images
    
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
    # Example usage and testing
    from ..utils.logger import setup_logger
    
    setup_logger("docling_parser", "logs/parser.log", logging.DEBUG)
    
    # Initialize parser
    parser = DoclingParser(use_ocr=True, extract_images=True)
    
    # Parse the factsheet
    pdf_path = "bajaj_finserv_factsheet_Oct.pdf"
    
    try:
        parsed_doc = parser.parse_pdf(pdf_path)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"PDF Parsing Summary: {parsed_doc.filename}")
        print(f"{'='*60}")
        print(f"Total Pages: {parsed_doc.total_pages}")
        print(f"Tables Extracted: {len(parsed_doc.tables)}")
        print(f"Text Sections: {len(parsed_doc.text_sections)}")
        print(f"Images/Charts: {len(parsed_doc.images)}")
        print(f"\nMetadata: {parsed_doc.metadata}")
        
        # Show sample table
        if parsed_doc.tables:
            print(f"\n{'='*60}")
            print("Sample Table (First Table):")
            print(f"{'='*60}")
            table = parsed_doc.tables[0]
            print(f"Table Name: {table.table_name}")
            print(f"Page: {table.page_number}")
            print(f"Dimensions: {table.rows} rows x {table.columns} columns")
            print(f"Headers: {table.headers}")
            print(f"\nMarkdown Preview:\n{table.markdown[:500]}...")
        
        # Save output
        parser.save_parsed_output(parsed_doc, "data/processed")
        print(f"\n✓ Parsed output saved to data/processed/")
        
    except Exception as e:
        logger.error(f"Error during parsing: {str(e)}", exc_info=True)
        print(f"\n✗ Error: {str(e)}")
