"""
SQLite Storage for Structured Data
Stores tables, metadata, and fund information for fast lookup
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class SQLiteStorage:
    """
    SQLite database for structured financial data
    
    Tables:
    - funds: Fund metadata (name, AUM, category, inception, etc.)
    - tables: Complete table data from PDF
    - performance: Performance metrics
    - holdings: Top holdings data
    - chunks_metadata: Chunk metadata for quick filtering
    """
    
    def __init__(self, db_path: str = "data/processed/factsheet.db"):
        """Initialize SQLite connection"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Allow connection to be used across threads (needed for Streamlit)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return dicts instead of tuples
        self.cursor = self.conn.cursor()
        
        self._create_tables()
        print(f"✓ Connected to SQLite database: {db_path}")
    
    def _create_tables(self):
        """Create database schema"""
        
        # Funds table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS funds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_name TEXT UNIQUE NOT NULL,
                category TEXT,
                aum_crores REAL,
                inception_date TEXT,
                benchmark TEXT,
                nav REAL,
                expense_ratio REAL,
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tables from PDF
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pdf_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_id TEXT UNIQUE NOT NULL,
                page_number INTEGER,
                fund_name TEXT,
                headers TEXT,  -- JSON array
                rows TEXT,     -- JSON array of arrays
                row_count INTEGER,
                col_count INTEGER,
                caption TEXT,
                dimensions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fund_name) REFERENCES funds(fund_name)
            )
        ''')
        
        # Performance metrics
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_name TEXT NOT NULL,
                metric_name TEXT,  -- e.g., "1 Year Return", "3 Year Return"
                value REAL,
                period TEXT,       -- e.g., "1Y", "3Y", "5Y"
                as_of_date TEXT,
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fund_name) REFERENCES funds(fund_name)
            )
        ''')
        
        # Holdings data
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_name TEXT NOT NULL,
                company_name TEXT,
                percentage REAL,
                sector TEXT,
                rank INTEGER,      -- Top 1, Top 2, etc.
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fund_name) REFERENCES funds(fund_name)
            )
        ''')
        
        # Chunk metadata for filtering
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunk_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT UNIQUE NOT NULL,
                chunk_type TEXT,   -- metadata, table, text, image
                page_number INTEGER,
                fund_name TEXT,
                content_preview TEXT,  -- First 200 chars
                word_count INTEGER,
                has_embedding BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for fast lookup
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_funds_name ON funds(fund_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_tables_fund ON pdf_tables(fund_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_fund ON performance(fund_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_holdings_fund ON holdings(fund_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunk_metadata(chunk_type)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunks_fund ON chunk_metadata(fund_name)')
        
        self.conn.commit()
    
    def insert_fund(self, fund_data: Dict[str, Any]) -> int:
        """Insert or update fund metadata"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO funds (
                fund_name, category, aum_crores, inception_date,
                benchmark, nav, expense_ratio, page_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fund_data.get('fund_name'),
            fund_data.get('category'),
            fund_data.get('aum_crores'),
            fund_data.get('inception_date'),
            fund_data.get('benchmark'),
            fund_data.get('nav'),
            fund_data.get('expense_ratio'),
            fund_data.get('page_number')
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_table(self, table_data: Dict[str, Any]) -> int:
        """Insert table data"""
        self.cursor.execute('''
            INSERT OR REPLACE INTO pdf_tables (
                table_id, page_number, fund_name, headers, rows,
                row_count, col_count, caption, dimensions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            table_data.get('table_id'),
            table_data.get('page_number'),
            table_data.get('fund_name'),
            json.dumps(table_data.get('headers', [])),
            json.dumps(table_data.get('rows', [])),
            table_data.get('row_count'),
            table_data.get('col_count'),
            table_data.get('caption'),
            table_data.get('dimensions')
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def insert_chunk_metadata(self, chunk: Dict[str, Any]) -> int:
        """Insert chunk metadata for filtering"""
        content_preview = chunk['content'][:200] if len(chunk['content']) > 200 else chunk['content']
        
        self.cursor.execute('''
            INSERT OR REPLACE INTO chunk_metadata (
                chunk_id, chunk_type, page_number, fund_name,
                content_preview, word_count, has_embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk['chunk_id'],
            chunk['chunk_type'],
            chunk['page_number'],
            chunk.get('fund_name'),
            content_preview,
            chunk.get('metadata', {}).get('word_count', 0),
            'embedding' in chunk
        ))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_fund(self, fund_name: str) -> Optional[Dict[str, Any]]:
        """Get fund metadata by name"""
        self.cursor.execute('SELECT * FROM funds WHERE fund_name = ?', (fund_name,))
        row = self.cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_funds(self) -> List[Dict[str, Any]]:
        """Get all funds"""
        self.cursor.execute('SELECT * FROM funds ORDER BY fund_name')
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_tables_by_fund(self, fund_name: str) -> List[Dict[str, Any]]:
        """Get all tables for a specific fund"""
        self.cursor.execute('''
            SELECT * FROM pdf_tables 
            WHERE fund_name = ?
            ORDER BY page_number
        ''', (fund_name,))
        
        tables = []
        for row in self.cursor.fetchall():
            table = dict(row)
            table['headers'] = json.loads(table['headers']) if table['headers'] else []
            table['rows'] = json.loads(table['rows']) if table['rows'] else []
            tables.append(table)
        
        return tables
    
    def get_chunks_by_type(self, chunk_type: str) -> List[Dict[str, Any]]:
        """Get all chunks of a specific type"""
        self.cursor.execute('''
            SELECT * FROM chunk_metadata 
            WHERE chunk_type = ?
            ORDER BY page_number
        ''', (chunk_type,))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_chunks_by_fund(self, fund_name: str) -> List[Dict[str, Any]]:
        """Get all chunks related to a specific fund"""
        self.cursor.execute('''
            SELECT * FROM chunk_metadata 
            WHERE fund_name = ?
            ORDER BY page_number
        ''', (fund_name,))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_funds(self, query: str) -> List[Dict[str, Any]]:
        """Search funds by name or category"""
        self.cursor.execute('''
            SELECT * FROM funds 
            WHERE fund_name LIKE ? OR category LIKE ?
            ORDER BY fund_name
        ''', (f'%{query}%', f'%{query}%'))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        stats = {}
        
        self.cursor.execute('SELECT COUNT(*) as count FROM funds')
        stats['total_funds'] = self.cursor.fetchone()['count']
        
        self.cursor.execute('SELECT COUNT(*) as count FROM pdf_tables')
        stats['total_tables'] = self.cursor.fetchone()['count']
        
        self.cursor.execute('SELECT COUNT(*) as count FROM chunk_metadata')
        stats['total_chunks'] = self.cursor.fetchone()['count']
        
        self.cursor.execute('SELECT COUNT(*) as count FROM chunk_metadata WHERE has_embedding = TRUE')
        stats['chunks_with_embeddings'] = self.cursor.fetchone()['count']
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def populate_from_chunks(db: SQLiteStorage, chunks_path: str):
    """Populate database from chunks file"""
    
    print(f"Loading chunks from: {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    chunks = chunks_data['chunks']
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Extract and insert fund metadata
    print("\nExtracting fund metadata...")
    fund_names = set()
    for chunk in chunks:
        if chunk.get('fund_name'):
            fund_names.add(chunk['fund_name'])
        
        # Extract from metadata chunks
        if chunk['chunk_type'] == 'metadata' and chunk.get('fund_name'):
            metadata = chunk.get('metadata', {})
            fund_data = {
                'fund_name': chunk['fund_name'],
                'category': metadata.get('Category'),
                'aum_crores': None,  # Parse from AUM string if needed
                'inception_date': metadata.get('Inception_Date'),
                'benchmark': metadata.get('Benchmark'),
                'nav': None,
                'expense_ratio': None,
                'page_number': chunk['page_number']
            }
            
            # Parse AUM if available
            aum_str = metadata.get('AUM', '')
            if aum_str:
                try:
                    # Remove '₹' and 'in Crore' and convert to float
                    aum_str = aum_str.replace('₹', '').replace(',', '').strip()
                    fund_data['aum_crores'] = float(aum_str)
                except:
                    pass
            
            db.insert_fund(fund_data)
    
    print(f"✓ Inserted {len(fund_names)} unique funds")
    
    # Insert table data
    print("\nInserting table chunks...")
    table_count = 0
    for chunk in chunks:
        if chunk['chunk_type'] == 'table':
            table_data = {
                'table_id': chunk['chunk_id'],
                'page_number': chunk['page_number'],
                'fund_name': chunk.get('fund_name'),
                'headers': chunk['metadata'].get('headers', []),
                'rows': [],  # Rows are in content as markdown
                'row_count': chunk['metadata'].get('row_count', 0),
                'col_count': chunk['metadata'].get('col_count', 0),
                'caption': chunk['metadata'].get('caption'),
                'dimensions': chunk['metadata'].get('dimensions')
            }
            db.insert_table(table_data)
            table_count += 1
    
    print(f"✓ Inserted {table_count} tables")
    
    # Insert chunk metadata
    print("\nInserting chunk metadata...")
    for chunk in chunks:
        db.insert_chunk_metadata(chunk)
    
    print(f"✓ Inserted {len(chunks)} chunk metadata records")


def main():
    """Test SQLite storage"""
    
    db_path = "data/processed/factsheet.db"
    
    # Remove existing database for clean test
    if Path(db_path).exists():
        Path(db_path).unlink()
        print(f"Removed existing database: {db_path}\n")
    
    # Create database and populate
    with SQLiteStorage(db_path) as db:
        # Populate from chunks
        chunks_path = "data/processed/chunks.json"
        populate_from_chunks(db, chunks_path)
        
        # Display statistics
        print(f"\n{'='*70}")
        print("DATABASE STATISTICS")
        print(f"{'='*70}\n")
        
        stats = db.get_statistics()
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Test queries
        print(f"\n{'='*70}")
        print("SAMPLE QUERIES")
        print(f"{'='*70}\n")
        
        # Get all funds
        funds = db.get_all_funds()
        print(f"All funds ({len(funds)}):")
        for fund in funds[:5]:
            print(f"  • {fund['fund_name']}")
            print(f"    Category: {fund['category']}")
            print(f"    Page: {fund['page_number']}")
        
        if len(funds) > 5:
            print(f"  ... and {len(funds) - 5} more")
        
        # Search funds
        print(f"\nSearch 'Large Cap':")
        results = db.search_funds('Large Cap')
        for fund in results:
            print(f"  • {fund['fund_name']} - {fund['category']}")
        
        # Get chunks by type
        print(f"\nMetadata chunks:")
        meta_chunks = db.get_chunks_by_type('metadata')
        print(f"  Total: {len(meta_chunks)}")
        for chunk in meta_chunks[:3]:
            print(f"  • {chunk['chunk_id']} - Page {chunk['page_number']}")
        
        # Get chunks for specific fund
        if funds:
            test_fund = funds[0]['fund_name']
            print(f"\nChunks for '{test_fund}':")
            fund_chunks = db.get_chunks_by_fund(test_fund)
            print(f"  Total chunks: {len(fund_chunks)}")
            for ctype in ['metadata', 'table', 'text']:
                count = sum(1 for c in fund_chunks if c['chunk_type'] == ctype)
                print(f"    {ctype}: {count}")
    
    print(f"\n{'='*70}")
    print("✅ SQLite Storage Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
