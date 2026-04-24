"""
Multi-modal RAG System using LlamaIndex

This module implements a Retrieval Augmented Generation system that can handle
text, table, image, and code content. It uses LlamaIndex for vector storage and retrieval,
and integrates with specialized agents for different content types.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import json
import base64
from io import BytesIO
import sqlite3
import tempfile
import os

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from docling_parsing import BaseChunk, TextChunk, TableChunk, ImageChunk, CodeChunk, ChunkType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableQueryAgent:
    """
    Agent for handling table queries using text-to-SQL approach.
    Converts natural language queries to SQL and executes them on table data.
    """
    
    def __init__(self, llm_model_name: str = "microsoft/DialoGPT-small"):
        """Initialize the table query agent."""
        self.llm_model_name = llm_model_name
        self.temp_db_path = None
        
        # Initialize LLM for SQL generation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
            logger.info(f"Table agent initialized with model: {llm_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load LLM for table agent: {e}")
            self.tokenizer = None
            self.model = None
    
    def query_table(self, table_chunk: TableChunk, query: str) -> str:
        """
        Query a table using natural language.
        
        Args:
            table_chunk: TableChunk containing the table data
            query: Natural language query
            
        Returns:
            Query result as formatted string
        """
        try:
            # Create temporary SQLite database
            self._create_temp_table(table_chunk)
            
            # Generate SQL query from natural language
            sql_query = self._generate_sql_query(table_chunk, query)
            
            # Execute SQL query
            result = self._execute_sql_query(sql_query)
            
            # Clean up
            self._cleanup_temp_db()
            
            return result
            
        except Exception as e:
            logger.error(f"Table query failed: {e}")
            # Fallback to simple table description
            return self._get_table_summary(table_chunk, query)
    
    def _create_temp_table(self, table_chunk: TableChunk):
        """Create a temporary SQLite table from chunk data."""
        self.temp_db_path = tempfile.mktemp(suffix='.db')
        conn = sqlite3.connect(self.temp_db_path)
        
        try:
            # Create DataFrame
            df = pd.DataFrame(table_chunk.table_data, columns=table_chunk.headers)
            
            # Save to SQLite
            df.to_sql('query_table', conn, index=False, if_exists='replace')
            conn.commit()
            
        except Exception as e:
            logger.warning(f"Failed to create temp table: {e}")
            raise
        finally:
            conn.close()
    
    def _generate_sql_query(self, table_chunk: TableChunk, query: str) -> str:
        """Generate SQL query from natural language query."""
        if not self.model or not self.tokenizer:
            # Fallback to simple SELECT
            return "SELECT * FROM query_table LIMIT 5;"
        
        try:
            # Create prompt for SQL generation
            schema_info = f"Table schema: {', '.join(table_chunk.headers)}"
            prompt = f"""Convert this question to SQL query:
Table: query_table
{schema_info}
Question: {query}
SQL:"""
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql_query = response[len(prompt):].strip()
            
            # Basic SQL validation and cleanup
            if not sql_query.upper().startswith('SELECT'):
                sql_query = "SELECT * FROM query_table LIMIT 5;"
                
            return sql_query
            
        except Exception as e:
            logger.warning(f"SQL generation failed: {e}")
            return "SELECT * FROM query_table LIMIT 5;"
    
    def _execute_sql_query(self, sql_query: str) -> str:
        """Execute SQL query and return formatted results."""
        conn = sqlite3.connect(self.temp_db_path)
        
        try:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            if not results:
                return "No results found for the query."
            
            # Format results
            formatted_result = f"Query Results ({len(results)} rows):\n"
            formatted_result += " | ".join(columns) + "\n"
            formatted_result += "-" * (len(" | ".join(columns))) + "\n"
            
            for row in results[:10]:  # Limit to 10 rows for context
                formatted_result += " | ".join(map(str, row)) + "\n"
            
            if len(results) > 10:
                formatted_result += f"... and {len(results) - 10} more rows\n"
                
            return formatted_result
            
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            return f"Query execution failed: {str(e)}"
        finally:
            conn.close()
    
    def _get_table_summary(self, table_chunk: TableChunk, query: str) -> str:
        """Fallback method to provide table summary."""
        summary = f"Table Summary:\n"
        summary += f"- Rows: {table_chunk.num_rows}\n"
        summary += f"- Columns: {table_chunk.num_cols}\n"
        summary += f"- Headers: {', '.join(table_chunk.headers)}\n"
        
        if table_chunk.table_data:
            summary += f"- Sample data: {table_chunk.table_data[0]}\n"
        
        return summary
    
    def _cleanup_temp_db(self):
        """Clean up temporary database file."""
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            try:
                os.unlink(self.temp_db_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp db: {e}")


class MultiModalRAG:
    """
    Multi-modal Retrieval Augmented Generation system using LlamaIndex.
    
    Handles different content types (text, tables, images, code) with specialized
    processing and uses embedding-based retrieval for finding relevant content.
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "microsoft/DialoGPT-small",
                 device: str = "auto"):
        """
        Initialize the Multi-modal RAG system.
        
        Args:
            embedding_model_name: HuggingFace model name for embeddings
            llm_model_name: HuggingFace model name for text generation
            device: Device to run models on
        """
        self.device = self._get_device(device)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            device=self.device
        )
        
        # Set global embedding model for LlamaIndex
        Settings.embed_model = self.embedding_model
        
        # Initialize LLM for response generation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
            self.llm_model.to(self.device)
            logger.info(f"Loaded LLM model: {llm_model_name}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
        
        # Initialize specialized agents
        self.table_agent = TableQueryAgent(llm_model_name)
        
        # Storage for chunks and index
        self.chunks: List[BaseChunk] = []
        self.vector_index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[VectorIndexRetriever] = None
        
        logger.info("Multi-modal RAG system initialized")
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for model inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def ingest_chunks(self, chunks: List[BaseChunk]):
        """
        Ingest chunks into the RAG system and create vector index.
        
        Args:
            chunks: List of parsed chunks from document
        """
        logger.info(f"Ingesting {len(chunks)} chunks into RAG system")
        
        self.chunks = chunks
        
        # Create LlamaIndex documents from chunk descriptions
        documents = []
        for chunk in chunks:
            # Use description as the main text for embedding
            document = Document(
                text=chunk.description,
                metadata={
                    'chunk_id': chunk.chunk_id,
                    'chunk_type': chunk.chunk_type.value,
                    'source_page': chunk.source_page,
                    # 'bbox': chunk.bbox,
                    # Store actual content based on chunk type
                    'content': self._serialize_chunk_content(chunk)
                }
            )
            documents.append(document)
        
        # Create vector store and index
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.vector_index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )
        
        # Create retriever
        self.retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=3
        )
        
        logger.info("Chunks successfully ingested and indexed")
    
    def _serialize_chunk_content(self, chunk: BaseChunk) -> Dict[str, Any]:
        """Serialize chunk content for metadata storage."""
        if isinstance(chunk, TextChunk):
            return {'text_content': chunk.text_content}
        elif isinstance(chunk, TableChunk):
            return {
                'table_data': chunk.table_data,
                'headers': chunk.headers,
                'table_html': chunk.table_html
            }
        elif isinstance(chunk, ImageChunk):
            return {
                'image_base64': chunk.image_base64,
                'image_format': chunk.image_format,
                'width': chunk.width,
                'height': chunk.height
            }
        elif isinstance(chunk, CodeChunk):
            return {
                'code_content': chunk.code_content,
                'programming_language': chunk.programming_language
            }
        else:
            return {}
    
    def query(self, query_text: str) -> str:
        """
        Process a query and return the generated response.
        
        Args:
            query_text: Natural language query
            
        Returns:
            Generated response based on retrieved context
        """
        if not self.retriever:
            raise ValueError("No chunks have been ingested. Call ingest_chunks() first.")
        
        logger.info(f"Processing query: {query_text}")
        
        # Retrieve relevant chunks
        retrieved_nodes = self.retriever.retrieve(query_text)
        
        # Process retrieved content based on chunk types
        context_parts = []
        for node_with_score in retrieved_nodes:
            context = self._process_retrieved_node(node_with_score, query_text)
            if context:
                context_parts.append(context)
        
        # Combine all context
        full_context = "\n\n".join(context_parts)
        
        # Generate response using LLM
        response = self._generate_response(query_text, full_context)
        
        logger.info("Query processing completed")
        return response
    
    def _process_retrieved_node(self, node_with_score: NodeWithScore, query: str) -> str:
        """
        Process a retrieved node based on its chunk type.
        
        Args:
            node_with_score: Retrieved node with similarity score
            query: Original query text
            
        Returns:
            Processed content as context string
        """
        node = node_with_score.node
        metadata = node.metadata
        chunk_type = metadata.get('chunk_type')
        content = metadata.get('content', {})
        
        logger.debug(f"Processing {chunk_type} chunk with score: {node_with_score.score}")
        
        try:
            if chunk_type == ChunkType.TEXT.value:
                # For text chunks, return the actual text content
                return f"Text Content:\n{content.get('text_content', '')}"
            
            elif chunk_type == ChunkType.TABLE.value:
                # For table chunks, use table query agent
                table_chunk = self._reconstruct_table_chunk(metadata, content)
                table_result = self.table_agent.query_table(table_chunk, query)
                return f"Table Data:\n{table_result}"
            
            elif chunk_type == ChunkType.IMAGE.value:
                # For image chunks, return description and metadata
                image_info = f"Image Description: {node.text}\n"
                if content.get('width') and content.get('height'):
                    image_info += f"Dimensions: {content['width']}x{content['height']}\n"
                if content.get('image_format'):
                    image_info += f"Format: {content['image_format']}\n"
                return f"Image Content:\n{image_info}"
            
            elif chunk_type == ChunkType.CODE.value:
                # For code chunks, return the actual code
                code_content = content.get('code_content', '')
                language = content.get('programming_language', 'unknown')
                return f"Code Content ({language}):\n```{language}\n{code_content}\n```"
            
        except Exception as e:
            logger.warning(f"Error processing {chunk_type} chunk: {e}")
            return f"Content Description: {node.text}"
        
        return f"Content Description: {node.text}"
    
    def _reconstruct_table_chunk(self, metadata: Dict, content: Dict) -> TableChunk:
        """Reconstruct TableChunk from metadata and content."""
        return TableChunk(
            chunk_id=metadata['chunk_id'],
            table_data=content.get('table_data', []),
            headers=content.get('headers', []),
            table_html=content.get('table_html'),
            num_rows=len(content.get('table_data', [])),
            num_cols=len(content.get('headers', [])),
            source_page=metadata.get('source_page'),
            # bbox=metadata.get('bbox')
        )
    
    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate response using LLM with retrieved context.
        
        Args:
            query: Original user query
            context: Retrieved and processed context
            
        Returns:
            Generated response
        """
        # Create prompt for response generation
        prompt = f"""Based on the following context, answer the user's question comprehensively and accurately.

Context:
{context}

Question: {query}

Answer:"""

        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=2048,
                truncation=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            
            # Clean up response
            if not response or len(response) < 10:
                response = self._generate_fallback_response(query, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_response(query, context)
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when LLM fails."""
        if not context.strip():
            return "I couldn't find relevant information to answer your question."
        
        # Simple extractive summary
        context_lines = context.split('\n')
        relevant_lines = [line for line in context_lines if line.strip()][:5]
        
        response = f"Based on the available information:\n\n"
        response += '\n'.join(relevant_lines)
        
        return response
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[BaseChunk]:
        """Retrieve a specific chunk by its ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[BaseChunk]:
        """Retrieve all chunks of a specific type."""
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about ingested chunks."""
        if not self.chunks:
            return {"total_chunks": 0}
        
        type_counts = {}
        for chunk in self.chunks:
            chunk_type = chunk.chunk_type.value
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "chunk_types": type_counts,
            "indexed": self.vector_index is not None
        }


class RAGPipeline:
    """
    Complete RAG pipeline that combines document parsing and retrieval.
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model: str = "microsoft/DialoGPT-small"):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            embedding_model: Model name for embeddings
            llm_model: Model name for text generation
        """
        from docling_parsing import DoclingParser
        
        self.parser = DoclingParser()
        self.rag_system = MultiModalRAG(
            embedding_model_name=embedding_model,
            llm_model_name=llm_model
        )
        
        logger.info("RAG Pipeline initialized")
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document end-to-end: parse, ingest, and prepare for querying.
        
        Args:
            document_path: Path to document file
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing document: {document_path}")
        
        # Parse document
        chunks = self.parser.parse_document(document_path)
        
        # Ingest into RAG system
        self.rag_system.ingest_chunks(chunks)
        
        # Return statistics
        stats = self.rag_system.get_statistics()
        logger.info(f"Document processing completed: {stats}")
        
        return stats
    
    def query_document(self, query: str) -> str:
        """
        Query the processed document.
        
        Args:
            query: Natural language query
            
        Returns:
            Generated response
        """
        return self.rag_system.query(query)
    
    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the processed document."""
        return self.rag_system.get_statistics()


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize RAG pipeline
        pipeline = RAGPipeline()
        
        # Process a document
        document_path = "data/SampleReport.pdf"
        if os.path.exists(document_path):
            stats = pipeline.process_document(document_path)
            print(f"Document processed successfully: {stats}")
            
            # Example queries
            queries = [
                "What is this document about?",
                "Show me any tables in the document",
                "What images are included?",
                "Is there any code in the document?"
            ]
            
            for query in queries:
                print(f"\nQuery: {query}")
                response = pipeline.query_document(query)
                print(f"Response: {response}")
                
        else:
            print("Please provide a valid document path for testing.")
            
            # Demonstrate with mock data
            from docling_parsing import TextChunk, TableChunk
            
            mock_chunks = [
                TextChunk(
                    chunk_id="text_1",
                    text_content="This is a sample text about machine learning.",
                    description="Text discussing machine learning concepts and applications."
                ),
                TableChunk(
                    chunk_id="table_1",
                    table_data=[["Alice", 25], ["Bob", 30]],
                    headers=["Name", "Age"],
                    description="Table containing person names and their ages."
                )
            ]
            
            # Test RAG system directly
            rag = MultiModalRAG()
            rag.ingest_chunks(mock_chunks)
            
            response = rag.query("Tell me about the people in the data")
            print(f"Mock data response: {response}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
