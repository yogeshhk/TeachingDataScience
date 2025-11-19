"""
Production PDF Parser using PyMuPDF
Extracts text, tables, and images from financial documents
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF
import json


@dataclass
class ParsedTable:
    """Structured table data"""
    table_name: Optional[str] = None
    page_number: int = 0
    table_index: int = 0
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1)
    caption: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert table to markdown format"""
        if not self.headers or not self.rows:
            return ""
        
        # Header row
        md = "| " + " | ".join(str(h) for h in self.headers) + " |\n"
        # Separator
        md += "| " + " | ".join("---" for _ in self.headers) + " |\n"
        # Data rows
        for row in self.rows:
            md += "| " + " | ".join(str(cell) for cell in row) + " |\n"
        
        return md


@dataclass
class ParsedImage:
    """Structured image data"""
    page_number: int = 0
    image_index: int = 0
    width: int = 0
    height: int = 0
    format: str = ""  # png, jpeg
    size_bytes: int = 0
    bbox: Optional[tuple] = None
    xref: int = 0  # PDF internal reference
    image_type: str = "unknown"  # chart, logo, diagram, decorative
    description: Optional[str] = None  # From Vision API later
    caption: Optional[str] = None


@dataclass
class ParsedText:
    """Text section with metadata"""
    content: str
    page_number: int
    section: Optional[str] = None
    char_count: int = 0


@dataclass
class ParsedDocument:
    """Complete parsed document"""
    filename: str
    total_pages: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[ParsedTable] = field(default_factory=list)
    text_sections: List[ParsedText] = field(default_factory=list)
    images: List[ParsedImage] = field(default_factory=list)
    parse_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "filename": self.filename,
            "total_pages": self.total_pages,
            "metadata": self.metadata,
            "tables": [
                {
                    "table_name": t.table_name,
                    "page": t.page_number,
                    "headers": t.headers,
                    "rows": t.rows,
                    "dimensions": f"{t.row_count}x{t.col_count}",
                    "caption": t.caption
                }
                for t in self.tables
            ],
            "images": [
                {
                    "page_number": img.page_number,
                    "image_type": img.image_type,
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "size_bytes": img.size_bytes,
                    "xref": img.xref,
                    "description": img.description,
                    "caption": img.caption
                }
                for img in self.images
            ],
            "text_sections": [
                {
                    "page": txt.page_number,
                    "section": txt.section,
                    "char_count": txt.char_count,
                    "content": txt.content  # Full content for chunking
                }
                for txt in self.text_sections
            ],
            "parse_timestamp": self.parse_timestamp
        }


class PDFParser:
    """
    Production PDF parser for financial documents
    Uses PyMuPDF for fast, reliable extraction
    """
    
    def __init__(self, min_table_rows: int = 2, min_image_size: int = 10000):
        """
        Initialize parser
        
        Args:
            min_table_rows: Minimum rows to consider as table
            min_image_size: Minimum size (bytes) to extract image
        """
        self.min_table_rows = min_table_rows
        self.min_image_size = min_image_size
    
    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse PDF and extract all content
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ParsedDocument with all extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        result = ParsedDocument(
            filename=pdf_path.name,
            total_pages=len(doc),
            metadata=self._extract_metadata(doc)
        )
        
        # Extract content page by page
        for page_num, page in enumerate(doc, 1):
            # Extract text
            text_content = self._extract_text(page, page_num)
            if text_content:
                result.text_sections.append(text_content)
            
            # Extract tables
            tables = self._extract_tables(page, page_num)
            result.tables.extend(tables)
            
            # Extract images
            images = self._extract_images(page, page_num, doc)
            result.images.extend(images)
        
        doc.close()
        return result
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract PDF metadata"""
        metadata = doc.metadata or {}
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "mod_date": metadata.get("modDate", ""),
        }
    
    def _extract_text(self, page: fitz.Page, page_num: int) -> ParsedText:
        """Extract text from page"""
        text = page.get_text()
        return ParsedText(
            content=text,
            page_number=page_num,
            char_count=len(text)
        )
    
    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[ParsedTable]:
        """Extract tables from page"""
        tables = []
        
        try:
            table_finder = page.find_tables()
            if not table_finder or not table_finder.tables:
                return tables
            
            for idx, table in enumerate(table_finder.tables):
                # Extract table data
                table_data = table.extract()
                
                if not table_data or len(table_data) < self.min_table_rows:
                    continue
                
                # Assume first row is headers
                headers = table_data[0] if table_data else []
                rows = table_data[1:] if len(table_data) > 1 else []
                
                parsed_table = ParsedTable(
                    page_number=page_num,
                    table_index=idx,
                    headers=headers,
                    rows=rows,
                    row_count=len(table_data),
                    col_count=len(headers) if headers else 0,
                    bbox=table.bbox
                )
                
                tables.append(parsed_table)
        
        except Exception as e:
            print(f"Warning: Table extraction failed on page {page_num}: {e}")
        
        return tables
    
    def _extract_images(self, page: fitz.Page, page_num: int, doc: fitz.Document) -> List[ParsedImage]:
        """Extract images from page"""
        images = []
        
        try:
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                
                try:
                    # Extract image data
                    base_image = doc.extract_image(xref)
                    
                    width = base_image["width"]
                    height = base_image["height"]
                    size_bytes = len(base_image["image"])
                    
                    # Filter small/decorative images
                    if size_bytes < self.min_image_size:
                        continue
                    
                    # Classify image type
                    image_type = self._classify_image(width, height, size_bytes)
                    
                    parsed_image = ParsedImage(
                        page_number=page_num,
                        image_index=img_index,
                        width=width,
                        height=height,
                        format=base_image["ext"],
                        size_bytes=size_bytes,
                        xref=xref,
                        image_type=image_type
                    )
                    
                    images.append(parsed_image)
                
                except Exception as e:
                    print(f"Warning: Image {img_index} extraction failed on page {page_num}: {e}")
        
        except Exception as e:
            print(f"Warning: Image extraction failed on page {page_num}: {e}")
        
        return images
    
    def _classify_image(self, width: int, height: int, size_bytes: int) -> str:
        """Classify image type based on dimensions and size"""
        size_kb = size_bytes / 1024
        
        # Large images - likely charts or full-page graphics
        if size_kb > 500:
            return "chart_or_diagram"
        
        # Medium images - likely charts or icons
        if size_kb > 50:
            return "chart"
        
        # Small images - likely logos or icons
        if size_kb > 10:
            return "logo_or_icon"
        
        return "decorative"
    
    def save_results(self, result: ParsedDocument, output_path: str):
        """Save parsed results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved results to: {output_path}")


if __name__ == "__main__":
    # Test the parser
    parser = PDFParser()
    
    pdf_path = "bajaj_finserv_factsheet_Oct.pdf"
    print(f"\nParsing: {pdf_path}")
    print("=" * 70)
    
    result = parser.parse(pdf_path)
    
    print(f"\n✓ Parsed successfully!")
    print(f"  Pages: {result.total_pages}")
    print(f"  Tables: {len(result.tables)}")
    print(f"  Images: {len(result.images)}")
    print(f"  Text sections: {len(result.text_sections)}")
    
    # Save results
    output_path = "data/processed/parsed_document.json"
    parser.save_results(result, output_path)
    
    # Print sample table
    if result.tables:
        print(f"\nSample table (Page {result.tables[0].page_number}):")
        print(result.tables[0].to_markdown()[:500])
    
    # Print sample images
    if result.images:
        print(f"\nSample images:")
        for img in result.images[:5]:
            print(f"  Page {img.page_number}: {img.width}×{img.height}px "
                  f"({img.format}, {img.size_bytes/1024:.1f} KB) - {img.image_type}")
