"""
End-to-End Ingestion Pipeline
Processes PDF → Chunks → Embeddings → Storage
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.pdf_parser import PDFParser
from src.chunking.chunker import MultiTierChunker
from src.embeddings.embedder import EmbeddingGenerator
from src.storage.faiss_storage import FAISSStorage
from src.storage.sqlite_storage import SQLiteStorage
import json


class IngestionPipeline:
    """
    Complete ingestion pipeline for factsheet processing
    
    Steps:
    1. Parse PDF → Extract text, tables, images
    2. Chunk → Create optimized chunks
    3. Embed → Generate vector embeddings
    4. Store → Save to FAISS + SQLite
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        self.parser = PDFParser()
        self.chunker = MultiTierChunker()
        self.embedder = EmbeddingGenerator()
        self.vector_storage = None
        self.sql_storage = None
        
        print("✓ Ingestion pipeline initialized")
    
    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str = "data/processed"
    ):
        """
        Process a PDF through the complete pipeline
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory for outputs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"INGESTION PIPELINE: {Path(pdf_path).name}")
        print(f"{'='*70}\n")
        
        # Step 1: Parse PDF
        print("Step 1: Parsing PDF...")
        parsed_doc = self.parser.parse(pdf_path)
        
        # Save parsed output
        parsed_output = output_path / "parsed_document.json"
        self.parser.save_results(parsed_doc, str(parsed_output))
        
        print(f"✓ Parsed {parsed_doc.total_pages} pages")
        print(f"  - Tables: {len(parsed_doc.tables)}")
        print(f"  - Images: {len(parsed_doc.images)}")
        print(f"  - Text sections: {len(parsed_doc.text_sections)}")
        
        # Step 2: Chunk document
        print("\nStep 2: Chunking document...")
        parsed_dict = parsed_doc.to_dict()
        chunks = self.chunker.chunk_document(parsed_dict)
        
        # Save chunks
        chunks_output = output_path / "chunks.json"
        self.chunker.save_chunks(chunks, str(chunks_output))
        
        chunk_types = self.chunker._count_chunk_types(chunks)
        print(f"✓ Created {len(chunks)} chunks")
        for ctype, count in chunk_types.items():
            print(f"  - {ctype}: {count}")
        
        # Step 3: Generate embeddings
        print("\nStep 3: Generating embeddings...")
        chunks_dict = [chunk.to_dict() for chunk in chunks]
        chunks_with_embeddings = self.embedder.generate_embeddings(chunks_dict)
        
        # Save embeddings
        embeddings_output = output_path / "chunks_with_embeddings.json"
        self.embedder.save_embeddings(chunks_with_embeddings, str(embeddings_output))
        
        print(f"✓ Generated {len(chunks_with_embeddings)} embeddings")
        print(f"  - Model: {self.embedder.model_name}")
        print(f"  - Dimension: {self.embedder.embedding_dim}")
        
        # Step 4: Store in vector database
        print("\nStep 4: Storing in vector database (FAISS)...")
        self.vector_storage = FAISSStorage(
            embedding_dim=self.embedder.embedding_dim,
            index_path=str(output_path / "faiss_index.bin"),
            metadata_path=str(output_path / "faiss_metadata.pkl")
        )
        
        self.vector_storage.add_chunks(chunks_with_embeddings)
        self.vector_storage.save()
        
        stats = self.vector_storage.get_stats()
        print(f"✓ Indexed {stats['total_chunks']} chunks")
        print(f"  - Unique funds: {stats['unique_funds']}")
        
        # Step 5: Store in SQL database
        print("\nStep 5: Storing in SQL database (SQLite)...")
        db_path = output_path / "factsheet.db"
        
        # Remove existing DB for clean insert
        if db_path.exists():
            db_path.unlink()
        
        self.sql_storage = SQLiteStorage(str(db_path))
        
        # Populate from chunks
        from src.storage.sqlite_storage import populate_from_chunks
        populate_from_chunks(self.sql_storage, str(chunks_output))
        
        sql_stats = self.sql_storage.get_statistics()
        print(f"✓ Stored {sql_stats['total_chunks']} chunk records")
        print(f"  - Funds: {sql_stats['total_funds']}")
        print(f"  - Tables: {sql_stats['total_tables']}")
        
        # Summary
        print(f"\n{'='*70}")
        print("INGESTION COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"Outputs saved to: {output_path}")
        print(f"  - Parsed PDF: parsed_document.json")
        print(f"  - Chunks: chunks.json")
        print(f"  - Embeddings: chunks_with_embeddings.json")
        print(f"  - Vector index: faiss_index.bin")
        print(f"  - SQL database: factsheet.db")
        
        print(f"\n{'='*70}")
        print("✅ Pipeline Successful!")
        print(f"{'='*70}")
        
        return {
            'parsed_doc': parsed_doc,
            'chunks': chunks,
            'embeddings': chunks_with_embeddings,
            'stats': {
                'total_pages': parsed_doc.total_pages,
                'total_chunks': len(chunks),
                'chunk_types': chunk_types,
                'total_funds': stats['unique_funds'],
                'embedding_dim': self.embedder.embedding_dim
            }
        }


def main():
    """Run ingestion pipeline on Bajaj PDF"""
    
    pdf_path = "data/raw/bajaj_finserv_factsheet_Oct.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        print("Please place the PDF in data/raw/ directory")
        return
    
    # Create pipeline
    pipeline = IngestionPipeline()
    
    # Process PDF
    result = pipeline.process_pdf(pdf_path)
    
    # Display final stats
    print("\nFinal Statistics:")
    for key, value in result['stats'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
