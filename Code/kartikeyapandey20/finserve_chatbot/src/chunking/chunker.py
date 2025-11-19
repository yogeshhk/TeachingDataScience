"""
Multi-Tier Chunking Strategy for Financial Documents

This module implements a 4-tier chunking approach optimized for RAG:
- Tier 1: Metadata chunks (NAV, AUM, dates, fund info)
- Tier 2: Table chunks (preserve entire tables, never split)
- Tier 3: Text chunks (semantic paragraphs with context)
- Tier 4: Image chunks (with descriptions for semantic search)

Each chunk maintains source traceability (page number, fund name, chunk type).
"""

import json
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path


@dataclass
class Chunk:
    """Represents a single chunk of content for RAG system"""
    chunk_id: str
    chunk_type: str  # 'metadata', 'table', 'text', 'image'
    content: str  # Main text content for embedding
    page_number: int
    fund_name: Optional[str] = None
    metadata: Dict[str, Any] = None
    source_file: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class MultiTierChunker:
    """
    Intelligent chunker that creates optimized chunks for financial factsheets
    
    Design principles:
    1. Never split tables across chunks
    2. Preserve fund context in each chunk
    3. Create metadata chunks for quick numerical lookups
    4. Keep text chunks semantically coherent
    """
    
    def __init__(
        self,
        max_text_chunk_size: int = 512,  # tokens (roughly 400 words)
        overlap: int = 50,  # token overlap between text chunks
        extract_fund_names: bool = True
    ):
        self.max_text_chunk_size = max_text_chunk_size
        self.overlap = overlap
        self.extract_fund_names = extract_fund_names
        
        # Fund name patterns
        self.fund_patterns = [
            r'Bajaj Finserv (?:Large Cap|Flexi Cap|Multi Cap|Large and Mid Cap|Small Cap|'
            r'Consumption|Healthcare|Equity Savings|ELSS Tax Saver|Balanced Advantage|'
            r'Multi Asset Allocation|Arbitrage|Liquid|Money Market|Gilt|Overnight|'
            r'Banking and PSU|Nifty 50 ETF|Nifty Bank ETF|Nifty 1D Rate Liquid ETF|'
            r'Nifty 50 Index|Nifty Next 50 Index) Fund',
        ]
    
    def chunk_document(self, parsed_doc: dict) -> List[Chunk]:
        """
        Main entry point - chunks entire parsed document
        
        Args:
            parsed_doc: Dictionary from PDFParser.to_dict()
            
        Returns:
            List of Chunk objects ready for embedding
        """
        chunks = []
        source_file = parsed_doc.get('filename', 'unknown.pdf')
        
        # Tier 1: Metadata chunks (document-level and per-fund)
        chunks.extend(self._create_metadata_chunks(parsed_doc, source_file))
        
        # Tier 2: Table chunks (one chunk per table, never split)
        chunks.extend(self._create_table_chunks(parsed_doc, source_file))
        
        # Tier 3: Text chunks (semantic paragraphs with overlap)
        chunks.extend(self._create_text_chunks(parsed_doc, source_file))
        
        # Tier 4: Image chunks (with metadata for Vision API)
        chunks.extend(self._create_image_chunks(parsed_doc, source_file))
        
        return chunks
    
    def _create_metadata_chunks(self, parsed_doc: dict, source_file: str) -> List[Chunk]:
        """
        Tier 1: Extract structured metadata as searchable chunks
        
        Examples:
        - "Bajaj Finserv Large Cap Fund has AUM of ₹1,610.77 crores as of Oct 2024"
        - "NAV: ₹12.45, Inception: 20-Aug-2024, Category: Large Cap Fund"
        """
        chunks = []
        
        # Document-level metadata
        doc_metadata = parsed_doc.get('metadata', {})
        if doc_metadata:
            content = f"Document: {parsed_doc.get('filename', 'Factsheet')}\n"
            content += f"Total Pages: {parsed_doc.get('total_pages', 0)}\n"
            content += f"Created: {doc_metadata.get('creation_date', 'N/A')}\n"
            content += f"Modified: {doc_metadata.get('mod_date', 'N/A')}"
            
            chunk = Chunk(
                chunk_id=f"meta_doc_001",
                chunk_type='metadata',
                content=content,
                page_number=1,
                metadata=doc_metadata,
                source_file=source_file
            )
            chunks.append(chunk)
        
        # Fund-specific metadata from tables (extract key metrics)
        # We'll look for tables with fund names and key metrics
        tables = parsed_doc.get('tables', [])
        fund_metadata = self._extract_fund_metadata_from_tables(tables)
        
        for idx, (fund_name, metrics) in enumerate(fund_metadata.items()):
            content = f"Fund: {fund_name}\n"
            for key, value in metrics.items():
                content += f"{key}: {value}\n"
            
            chunk = Chunk(
                chunk_id=f"meta_fund_{idx+1:03d}",
                chunk_type='metadata',
                content=content.strip(),
                page_number=metrics.get('page', 0),
                fund_name=fund_name,
                metadata=metrics,
                source_file=source_file
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_fund_metadata_from_tables(self, tables: List[dict]) -> Dict[str, dict]:
        """
        Extract structured fund information from tables
        
        Looks for tables containing:
        - Fund Name
        - AUM (₹ in Crore)
        - Category
        - Benchmark
        - Inception Date
        - NAV
        - Performance metrics (Beta, Sharpe ratio, etc.)
        """
        fund_metadata = {}
        
        for table in tables:
            rows = table.get('rows', [])
            headers = table.get('headers', [])
            page = table.get('page', 0)
            
            # Look for fund snapshot tables (contain fund names and metrics)
            for row in rows:
                # Check if row contains a fund name
                fund_name = None
                for cell in row:
                    if cell and isinstance(cell, str):
                        for pattern in self.fund_patterns:
                            match = re.search(pattern, cell)
                            if match:
                                fund_name = match.group(0)
                                break
                    if fund_name:
                        break
                
                if fund_name:
                    if fund_name not in fund_metadata:
                        fund_metadata[fund_name] = {'page': page}
                    
                    # Extract metrics from the row
                    for i, cell in enumerate(row):
                        if cell and isinstance(cell, str):
                            # AUM pattern
                            if 'AUM' in cell or '₹ in Crore' in cell or 'Crore' in cell:
                                if i + 1 < len(row) and row[i + 1]:
                                    fund_metadata[fund_name]['AUM'] = row[i + 1]
                            
                            # Inception Date
                            if 'Inception' in cell:
                                if i + 1 < len(row) and row[i + 1]:
                                    fund_metadata[fund_name]['Inception_Date'] = row[i + 1]
                            
                            # Category
                            if 'Category' in cell:
                                if i + 1 < len(row) and row[i + 1]:
                                    fund_metadata[fund_name]['Category'] = row[i + 1]
                            
                            # Benchmark
                            if 'Benchmark' in cell:
                                if i + 1 < len(row) and row[i + 1]:
                                    fund_metadata[fund_name]['Benchmark'] = row[i + 1]
        
        return fund_metadata
    
    def _create_table_chunks(self, parsed_doc: dict, source_file: str) -> List[Chunk]:
        """
        Tier 2: Create one chunk per table (never split tables)
        
        Tables are converted to markdown for better LLM understanding
        Each table chunk includes:
        - Table content in markdown format
        - Page number
        - Fund context (if detected)
        - Table caption/title
        """
        chunks = []
        tables = parsed_doc.get('tables', [])
        
        for idx, table in enumerate(tables):
            page = table.get('page', 0)
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            caption = table.get('caption', '')
            dimensions = table.get('dimensions', '')
            
            # Convert table to markdown
            markdown = self._table_to_markdown(headers, rows, caption)
            
            # Detect fund name in table content
            fund_name = self._extract_fund_name(markdown)
            
            # Create descriptive content
            content = f"Table from Page {page}"
            if caption:
                content += f" - {caption}"
            if dimensions:
                content += f" ({dimensions})"
            content += "\n\n" + markdown
            
            chunk = Chunk(
                chunk_id=f"table_{idx+1:03d}",
                chunk_type='table',
                content=content,
                page_number=page,
                fund_name=fund_name,
                metadata={
                    'headers': headers,
                    'row_count': len(rows),
                    'col_count': len(headers) if headers else 0,
                    'caption': caption,
                    'dimensions': dimensions
                },
                source_file=source_file
            )
            chunks.append(chunk)
        
        return chunks
    
    def _table_to_markdown(self, headers: List[str], rows: List[List[str]], caption: str = None) -> str:
        """Convert table to markdown format"""
        md = []
        
        if caption:
            md.append(f"**{caption}**\n")
        
        # Headers
        if headers:
            clean_headers = [str(h) if h is not None else '' for h in headers]
            md.append('| ' + ' | '.join(clean_headers) + ' |')
            md.append('|' + '|'.join(['---' for _ in headers]) + '|')
        
        # Rows
        for row in rows:
            clean_row = [str(cell) if cell is not None else '' for cell in row]
            md.append('| ' + ' | '.join(clean_row) + ' |')
        
        return '\n'.join(md)
    
    def _create_text_chunks(self, parsed_doc: dict, source_file: str) -> List[Chunk]:
        """
        Tier 3: Create semantic text chunks with overlap
        
        Strategy:
        1. Split by paragraphs (double newline)
        2. Group paragraphs to target chunk size
        3. Add overlap for context preservation
        4. Extract fund names for each chunk
        """
        chunks = []
        text_sections = parsed_doc.get('text_sections', [])
        
        chunk_counter = 1
        
        for section in text_sections:
            # Handle both 'content' and 'preview' keys
            content = section.get('content', section.get('preview', ''))
            page_num = section.get('page_number', section.get('page', 0))
            
            if not content or len(content.strip()) < 50:
                continue  # Skip very short sections
            
            # Split into paragraphs
            paragraphs = self._split_into_paragraphs(content)
            
            # Group paragraphs into chunks
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para.split())  # Rough word count
                
                if current_size + para_size > self.max_text_chunk_size and current_chunk:
                    # Create chunk from accumulated paragraphs
                    chunk_content = '\n\n'.join(current_chunk)
                    fund_name = self._extract_fund_name(chunk_content)
                    
                    chunk = Chunk(
                        chunk_id=f"text_{chunk_counter:03d}",
                        chunk_type='text',
                        content=chunk_content,
                        page_number=page_num,
                        fund_name=fund_name,
                        metadata={
                            'char_count': len(chunk_content),
                            'word_count': current_size
                        },
                        source_file=source_file
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                    
                    # Start new chunk with overlap
                    if self.overlap > 0 and len(current_chunk) > 1:
                        # Keep last paragraph for overlap
                        current_chunk = [current_chunk[-1], para]
                        current_size = len(current_chunk[-1].split()) + para_size
                    else:
                        current_chunk = [para]
                        current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size
            
            # Don't forget the last chunk
            if current_chunk:
                chunk_content = '\n\n'.join(current_chunk)
                fund_name = self._extract_fund_name(chunk_content)
                
                chunk = Chunk(
                    chunk_id=f"text_{chunk_counter:03d}",
                    chunk_type='text',
                    content=chunk_content,
                    page_number=page_num,
                    fund_name=fund_name,
                    metadata={
                        'char_count': len(chunk_content),
                        'word_count': current_size
                    },
                    source_file=source_file
                )
                chunks.append(chunk)
                chunk_counter += 1
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (double newline separator)"""
        # Split on double newline or single newline if text is dense
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If paragraphs are too large, split on single newline
        final_paragraphs = []
        for para in paragraphs:
            if len(para.split()) > self.max_text_chunk_size:
                # Split further on single newline
                sub_paras = para.split('\n')
                final_paragraphs.extend([sp.strip() for sp in sub_paras if sp.strip()])
            else:
                final_paragraphs.append(para)
        
        return final_paragraphs
    
    def _create_image_chunks(self, parsed_doc: dict, source_file: str) -> List[Chunk]:
        """
        Tier 4: Create chunks for images (charts, diagrams)
        
        For now, creates placeholder chunks with image metadata.
        In production, these would be enhanced with Vision API descriptions.
        """
        chunks = []
        images = parsed_doc.get('images', [])
        
        for idx, image in enumerate(images):
            page = image.get('page_number', 0)
            width = image.get('width', 0)
            height = image.get('height', 0)
            img_format = image.get('format', 'unknown')
            img_type = image.get('image_type', 'unknown')
            caption = image.get('caption', '')
            description = image.get('description', '')
            
            # Create descriptive content
            content = f"Image from Page {page}: {width}x{height}px {img_format} ({img_type})"
            if caption:
                content += f"\nCaption: {caption}"
            if description:
                content += f"\nDescription: {description}"
            else:
                # Placeholder for Vision API description
                content += f"\n[Image description to be generated by Vision API]"
            
            # Detect fund context from surrounding text
            fund_name = None  # Could be enhanced by checking nearby text chunks
            
            chunk = Chunk(
                chunk_id=f"image_{idx+1:03d}",
                chunk_type='image',
                content=content,
                page_number=page,
                fund_name=fund_name,
                metadata={
                    'width': width,
                    'height': height,
                    'format': img_format,
                    'image_type': img_type,
                    'size_kb': image.get('size_bytes', 0) / 1024,
                    'xref': image.get('xref', None)
                },
                source_file=source_file
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_fund_name(self, text: str) -> Optional[str]:
        """Extract fund name from text using regex patterns"""
        if not self.extract_fund_names:
            return None
        
        for pattern in self.fund_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return None
    
    def save_chunks(self, chunks: List[Chunk], output_path: str):
        """Save chunks to JSON file"""
        chunks_dict = [chunk.to_dict() for chunk in chunks]
        
        output = {
            'total_chunks': len(chunks),
            'chunk_types': self._count_chunk_types(chunks),
            'generated_at': datetime.now().isoformat(),
            'chunks': chunks_dict
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(chunks)} chunks to: {output_path}")
    
    def _count_chunk_types(self, chunks: List[Chunk]) -> dict:
        """Count chunks by type"""
        counts = {}
        for chunk in chunks:
            counts[chunk.chunk_type] = counts.get(chunk.chunk_type, 0) + 1
        return counts


def main():
    """Test the chunker on parsed document"""
    
    # Load parsed document
    parsed_path = "data/processed/parsed_document.json"
    
    print(f"Loading parsed document from: {parsed_path}")
    with open(parsed_path, 'r', encoding='utf-8') as f:
        parsed_doc = json.load(f)
    
    print(f"✓ Loaded document: {parsed_doc['filename']}")
    print(f"  Pages: {parsed_doc['total_pages']}")
    print(f"  Tables: {len(parsed_doc['tables'])}")
    print(f"  Images: {len(parsed_doc['images'])}")
    print(f"  Text sections: {len(parsed_doc['text_sections'])}")
    
    # Create chunker
    chunker = MultiTierChunker(
        max_text_chunk_size=512,  # ~400 words
        overlap=50,
        extract_fund_names=True
    )
    
    print("\nChunking document...")
    chunks = chunker.chunk_document(parsed_doc)
    
    # Display statistics
    print(f"\n{'='*70}")
    print("CHUNKING RESULTS")
    print(f"{'='*70}")
    print(f"\nTotal Chunks Created: {len(chunks)}")
    
    chunk_types = chunker._count_chunk_types(chunks)
    print("\nChunks by Type:")
    for chunk_type, count in chunk_types.items():
        print(f"  ✓ {chunk_type.capitalize()}: {count}")
    
    # Sample chunks from each type
    print(f"\n{'='*70}")
    print("SAMPLE CHUNKS")
    print(f"{'='*70}")
    
    for chunk_type in ['metadata', 'table', 'text', 'image']:
        sample = next((c for c in chunks if c.chunk_type == chunk_type), None)
        if sample:
            print(f"\n--- {chunk_type.upper()} CHUNK ---")
            print(f"ID: {sample.chunk_id}")
            print(f"Page: {sample.page_number}")
            print(f"Fund: {sample.fund_name or 'N/A'}")
            print(f"Content (first 300 chars):")
            print(sample.content[:300] + "..." if len(sample.content) > 300 else sample.content)
            print(f"Metadata: {sample.metadata}")
    
    # Save chunks
    output_path = "data/processed/chunks.json"
    chunker.save_chunks(chunks, output_path)
    
    print(f"\n{'='*70}")
    print("✅ Chunking Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
