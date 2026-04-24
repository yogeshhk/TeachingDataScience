"""
Multi-modal Document Parsing using Docling

This module provides a comprehensive document parsing solution that extracts and processes
different content modalities (text, tables, images, code) from documents using docling,
and generates semantic descriptions for each chunk using LLM/VLM models.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import base64
from io import BytesIO
import json # Ensure this is present
from pathlib import Path # ADD THIS IMPORT
import pandas as pd
from langchain_groq import ChatGroq
import os

from pydantic import BaseModel, Field
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
import torch
from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem, TableItem


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Enum for different types of content chunks."""
    TEXT = "text"
    TABLE = "table" 
    IMAGE = "image"
    CODE = "code"


class BaseChunk(BaseModel):
    """Base class for all chunk types."""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    chunk_type: ChunkType = Field(description="Type of content chunk")
    description: str = Field(default="", description="AI-generated description of the chunk content")
    source_page: Optional[int] = Field(default=None, description="Source page number")
    # bbox: Optional[Dict[str, float]] = Field(default=None, description="Bounding box coordinates")
    parent_heading: Optional[str] = Field(default=None, description="Text of the current section heading")

class TextChunk(BaseChunk):
    """Chunk containing text content."""
    text_content: str = Field(description="The actual text content")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    word_count: Optional[int] = Field(default=None, description="Number of words in text")


class TableChunk(BaseChunk):
    """Chunk containing table data."""
    table_data: List[List[str]] = Field(description="Table data as list of rows")
    headers: List[str] = Field(description="Column headers")
    table_html: Optional[str] = Field(default=None, description="HTML representation of table")
    chunk_type: ChunkType = Field(default=ChunkType.TABLE)
    num_rows: Optional[int] = Field(default=None, description="Number of rows")
    num_cols: Optional[int] = Field(default=None, description="Number of columns")


class ImageChunk(BaseChunk):
    """Chunk containing image data."""
    image_path: Optional[str] = Field(default=None, description="Path to saved image file")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image data")
    image_format: Optional[str] = Field(default=None, description="Image format (png, jpg, etc.)")
    chunk_type: ChunkType = Field(default=ChunkType.IMAGE)
    width: Optional[int] = Field(default=None, description="Image width")
    height: Optional[int] = Field(default=None, description="Image height")


class CodeChunk(BaseChunk):
    """Chunk containing code content."""
    code_content: str = Field(description="The actual code content")
    programming_language: Optional[str] = Field(default=None, description="Detected programming language")
    chunk_type: ChunkType = Field(default=ChunkType.CODE)
    line_count: Optional[int] = Field(default=None, description="Number of lines of code")


class DoclingParser:
    """
    Multi-modal document parser using docling for extraction and AI models for description generation.
    
    This class handles parsing of various document types and generates semantic descriptions
    for different content modalities using appropriate AI models.
    """
    
    def __init__(self, 
                 text_model_name: str = "gemma2-9b-it",  # Good for summarization, not "microsoft/DialoGPT-small",
                 vision_model_name: str = "microsoft/git-base",
                 device: str = "auto"):
        """
        Initialize the DoclingParser with AI models.
        
        Args:
            text_model_name: HuggingFace model name for text description generation
            vision_model_name: HuggingFace model name for image description generation
            device: Device to run models on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.converter = DocumentConverter()
        
        # Initialize text generation model
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            # self.text_model = AutoModelForCausalLM.from_pretrained(text_model_name)
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("No Groq API key provided. Text descriptions will use fallback.")
                self.text_model = None            
            else:
                self.text_model = ChatGroq(
                    model=text_model_name,
                    api_key=api_key,
                    temperature=0.3,
                    max_tokens=150
                )
                self.text_model.to(self.device)
                logger.info(f"Loaded text model: {text_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load text model: {e}. Using fallback.")
            self.text_tokenizer = None
            self.text_model = None
        
        # Initialize vision model
        try:
            self.vision_processor = AutoProcessor.from_pretrained(vision_model_name)
            self.vision_model = AutoModelForVision2Seq.from_pretrained(vision_model_name)
            self.vision_model.to(self.device)
            logger.info(f"Loaded vision model: {vision_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load vision model: {e}. Using fallback.")
            self.vision_processor = None
            self.vision_model = None
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for model inference."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _is_heading(self, item: NodeItem) -> bool:
            """
            Check if a Docling NodeItem is a document heading.
            
            This check often relies on the label attribute of the node.
            """
            # docling headers often have a specific label like DocItemLabel.HEADER
            # or DocItemLabel.TITLE. We'll use a pragmatic approach based on common
            # docling labels for structure.
            heading_bool = (hasattr(item, 'label') and 
                    item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE] and 
                    isinstance(item, TextItem))
            return heading_bool
            
    
    def populate_chunks_with_parents(self, document_path: str) -> List[Union[TextChunk, TableChunk, ImageChunk, CodeChunk]]:
            """
            Parse a document, extract all content modalities, and assign a hierarchical
            parent (section heading text) to each chunk based on reading order.
            
            Args:
                document_path: Path to the document file
                
            Returns:
                List of parsed chunks with parent heading assigned
            """
            logger.info(f"Starting document parsing: {document_path}")
            
            # Convert document using docling
            try:
                result: ConversionResult = self.converter.convert(document_path)
                document: DoclingDocument = result.document
                logger.info("Document conversion successful")
            except Exception as e:
                logger.error(f"Document conversion failed: {e}")
                raise
            
            chunks = []
            current_heading_text: Optional[str] = None
            chunk_counter = 0

            # Iterate the elements in reading order, including hierarchy level:
            for item, level in document.iterate_items():
                
                # 1. Check if the item is a new section header
                if self._is_heading(item):
                    if isinstance(item, TextItem) and item.text:
                        # Reset the current heading to the text of the new header
                        # TODO: no level wise consideration, as stack based push and pop logic needs to be considered
                        current_heading_text = item.text
                        logger.debug(f"New section heading detected (Level {level}): {current_heading_text}")
                    continue # Do not add the header itself to the chunks list

                # 2. Extract content chunks and assign the current_heading_text as parent
                chunk: Optional[BaseChunk] = None
                source_page = item.prov[0].page_no if item.prov else None
                # bbox = item.bbox.to_dict() if item.bbox else None
                
                if isinstance(item, TextItem):
                    # Ensure it's not a heading we missed or a very small artifact
                    if item.text and len(item.text.split()) > 1: 
                        chunk = TextChunk(
                            chunk_id=f"text_{chunk_counter}",
                            text_content=item.text,
                            word_count=len(item.text.split()),
                            source_page=source_page,
                            # bbox=bbox,
                            parent_heading=current_heading_text
                        )
                
                elif isinstance(item, TableItem):
                    # Using logic similar to _extract_table_chunks for data extraction
                    table_data = item.export_to_dataframe().values.tolist()
                    headers = list(item.export_to_dataframe().columns)
                    rows = table_data
                    
                    chunk = TableChunk(
                        chunk_id=f"table_{chunk_counter}",
                        table_data=rows,
                        headers=headers,
                        table_html=getattr(item, 'html', None),
                        num_rows=len(rows),
                        num_cols=len(headers),
                        source_page=source_page,
                        # bbox=bbox,
                        parent_heading=current_heading_text
                    )

                # NOTE: For images and code, the docling iteration may yield a NodeItem that
                # only *references* the actual image/code element stored in document.images/document.codes.
                # A full implementation would need to look up the item.ref in the respective list.
                # For simplicity in this structure, we'll continue with the Text/Table focus.
                
                if chunk:
                    # Assign the current heading text to the chunk (requires BaseChunk modification)
                    # chunk.parent_heading = current_heading_text 
                    chunks.append(chunk)
                    chunk_counter += 1
            
            logger.info(f"Extracted {len(chunks)} content chunks with hierarchical parent assignment.")
            
            # NOTE: Uncomment the description generation below for the full pipeline
            # chunks_with_descriptions = self._generate_descriptions(chunks)
            # return chunks_with_descriptions
            return chunks
        
    def _load_chunks(self, chunks_cache_path: str) -> List[Union[TextChunk, TableChunk, ImageChunk, CodeChunk]]:
        try:
            logger.info(f"Loading chunks from cache: {chunks_cache_path}")
            with open(chunks_cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                chunks = []
                for item in data:
                    # Deserialize using Pydantic model_validate based on chunk_type
                    chunk_type_enum = ChunkType(item['chunk_type'])
                    
                    # Use a robust switch for model validation
                    if chunk_type_enum == ChunkType.TEXT:
                        chunks.append(TextChunk.model_validate(item))
                    elif chunk_type_enum == ChunkType.TABLE:
                        chunks.append(TableChunk.model_validate(item))
                    elif chunk_type_enum == ChunkType.IMAGE:
                        chunks.append(ImageChunk.model_validate(item))
                    elif chunk_type_enum == ChunkType.CODE:
                        chunks.append(CodeChunk.model_validate(item))
                    # Ignore unknown chunk types for forward compatibility
                
                logger.info(f"Successfully loaded {len(chunks)} chunks from cache. Skipping fresh parsing.")
                return chunks

        except Exception as e:
            # Catches JSONDecodeError (like "Expecting value") or Pydantic validation errors
            logger.error(f"Failed to load chunks from cache: {e}. Proceeding with fresh parsing.")
            # IMPORTANT: We fall through to the fresh parsing logic below instead of returning.
                
    def _save_chunks(self, chunks: List[Union[TextChunk, TableChunk, ImageChunk, CodeChunk]]):
        try:
            # Define the cache directory and file path
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            chunks_cache_path = data_dir / "chunks.json"

            # Serialize all chunks into a list of dictionaries using Pydantic's model_dump()
            # Use mode='json' to properly serialize Enums and other non-standard types
            serialized_chunks = [chunk.model_dump(mode='json') for chunk in chunks]

            # Write the data to the JSON file
            with open(chunks_cache_path, 'w', encoding='utf-8') as f:
                json.dump(serialized_chunks, f, indent=4)
                
            logger.info(f"Successfully saved {len(chunks)} chunks to cache: {chunks_cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save chunks to cache: {e}")

        return chunks
    
    def parse_document(self, document_path: str) -> List[Union[TextChunk, TableChunk, ImageChunk, CodeChunk]]:
        """
        Parse a document and extract all content modalities.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            List of parsed chunks with descriptions
        """
        data_dir = Path("data")
        chunks_cache_path = data_dir / "chunks.json"
        if chunks_cache_path.exists():
            return self._load_chunks(chunks_cache_path)
            
        logger.info(f"Starting document parsing: {document_path}")
        
        # Convert document using docling
        try:
            result: ConversionResult = self.converter.convert(document_path)
            document = result.document
            logger.info("Document conversion successful")
        except Exception as e:
            logger.error(f"Document conversion failed: {e}")
            raise
        

        # for table_ix, table in enumerate(document.tables):
        #     page_number = table.prov[0].page_no if table.prov else "Unknown"
        #     table_df: pd.DataFrame = table.export_to_dataframe()
        #     print(f"## Table {table_ix} (Page {page_number})")
        #     print(table_df.to_markdown())

        # document.print_element_tree()

        # ## Iterate the elements in reading order, including hierarchy level:
        # for item, level in document.iterate_items():
        #     if isinstance(item, TextItem):
        #         print(item.text)
        #     elif isinstance(item, TableItem):
        #         table_df: pd.DataFrame = item.export_to_dataframe()
        #         print(table_df.to_markdown())
        
        chunks = self.populate_chunks_with_parents(document_path)

    
        # # Extract text chunks
        # text_chunks = self._extract_text_chunks(document)
        # chunks.extend(text_chunks)
        
        
        # # Extract table chunks
        # table_chunks = self._extract_table_chunks(document)
        # chunks.extend(table_chunks)
        
        # # Extract image chunks
        # image_chunks = self._extract_image_chunks(document)
        # chunks.extend(image_chunks)
        
        # # Extract code chunks
        # code_chunks = self._extract_code_chunks(document)
        # chunks.extend(code_chunks)
        
        # Generate descriptions for all chunks
        chunks_with_descriptions = self._generate_descriptions(chunks)
                
        self._save_chunks(chunks_with_descriptions)
        
        logger.info(f"Document parsing completed. Total chunks: {len(chunks_with_descriptions)}")
        return chunks_with_descriptions
    
    def _extract_text_chunks(self, document) -> List[TextChunk]:
        """Extract text content from document."""
        text_chunks = []
        
        try:
            # Extract text elements from docling document
            for i, text_elem in enumerate(document.texts):
                chunk = TextChunk(
                    chunk_id=f"text_{i}",
                    text_content=text_elem.text,
                    word_count=len(text_elem.text.split()),
                    source_page=getattr(text_elem, 'page', None),
                    # bbox=getattr(text_elem, 'bbox', None)
                )
                text_chunks.append(chunk)
            
            logger.info(f"Extracted {len(text_chunks)} text chunks")
        except Exception as e:
            logger.error(f"Error extracting text chunks: {e}")
        
        return text_chunks
    
    def _extract_table_chunks(self, document) -> List[TableChunk]:
        """Extract table content from document."""
        table_chunks = []
        
        try:
            # Extract table elements from docling document
            for i, table_elem in enumerate(document.tables):
                # Convert table to structured format
                headers = []
                rows = []
                
                if hasattr(table_elem, 'data'):
                    table_data = table_elem.data
                    if table_data and len(table_data) > 0:
                        headers = table_data[0] if table_data else []
                        rows = table_data[1:] if len(table_data) > 1 else []
                
                chunk = TableChunk(
                    chunk_id=f"table_{i}",
                    table_data=rows,
                    headers=headers,
                    table_html=getattr(table_elem, 'html', None),
                    num_rows=len(rows),
                    num_cols=len(headers),
                    source_page=getattr(table_elem, 'page', None),
                    # bbox=getattr(table_elem, 'bbox', None)
                )
                table_chunks.append(chunk)
            
            logger.info(f"Extracted {len(table_chunks)} table chunks")
        except Exception as e:
            logger.error(f"Error extracting table chunks: {e}")
        
        return table_chunks
    
    def _extract_image_chunks(self, document) -> List[ImageChunk]:
        """Extract image content from document."""
        image_chunks = []
        
        try:
            # Extract image elements from docling document
            for i, image_elem in enumerate(document.images):
                # Convert image to base64 if available
                image_base64 = None
                image_format = None
                
                if hasattr(image_elem, 'image') and image_elem.image:
                    try:
                        # Convert PIL Image to base64
                        buffered = BytesIO()
                        image_elem.image.save(buffered, format="PNG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode()
                        image_format = "png"
                    except Exception as e:
                        logger.warning(f"Failed to convert image to base64: {e}")
                
                chunk = ImageChunk(
                    chunk_id=f"image_{i}",
                    image_base64=image_base64,
                    image_format=image_format,
                    width=getattr(image_elem, 'width', None),
                    height=getattr(image_elem, 'height', None),
                    source_page=getattr(image_elem, 'page', None),
                    # bbox=getattr(image_elem, 'bbox', None)
                )
                image_chunks.append(chunk)
            
            logger.info(f"Extracted {len(image_chunks)} image chunks")
        except Exception as e:
            logger.error(f"Error extracting image chunks: {e}")
        
        return image_chunks
    
    def _extract_code_chunks(self, document) -> List[CodeChunk]:
        """Extract code content from document."""
        code_chunks = []
        
        try:
            # Extract code elements from docling document
            for i, code_elem in enumerate(document.codes):
                chunk = CodeChunk(
                    chunk_id=f"code_{i}",
                    code_content=code_elem.text,
                    programming_language=getattr(code_elem, 'language', None),
                    line_count=len(code_elem.text.splitlines()),
                    source_page=getattr(code_elem, 'page', None),
                    # bbox=getattr(code_elem, 'bbox', None)
                )
                code_chunks.append(chunk)
            
            logger.info(f"Extracted {len(code_chunks)} code chunks")
        except Exception as e:
            logger.error(f"Error extracting code chunks: {e}")
        
        return code_chunks
    
    def _generate_descriptions(self, chunks: List[BaseChunk]) -> List[BaseChunk]:
        """Generate AI descriptions for all chunks."""
        logger.info("Generating descriptions for chunks...")
        
        for chunk in chunks:
            try:
                if isinstance(chunk, TextChunk):
                    chunk.description = self._generate_text_description(chunk.text_content)
                elif isinstance(chunk, TableChunk):
                    chunk.description = self._generate_table_description(chunk)
                elif isinstance(chunk, ImageChunk):
                    chunk.description = self._generate_image_description(chunk)
                elif isinstance(chunk, CodeChunk):
                    chunk.description = self._generate_code_description(chunk.code_content, chunk.programming_language)
                    
                # for all chunks add their location ie parent heading in front so that it gets embedded also
                chunk.description = chunk.parent_heading + " : " + chunk.description
                
            except Exception as e:
                logger.warning(f"Failed to generate description for chunk {chunk.chunk_id}: {e}")
                chunk.description = f"Content chunk of type {chunk.chunk_type.value}"
        
        return chunks
    
    def _generate_text_description(self, text_content: str) -> str:
        """Generate description for text content using ChatGroq."""
        if not self.text_model:
            # Fallback: simple text summarization
            sentences = text_content.split('.')[:2]
            summary = ' '.join(sentences)
            return f"{summary[:150]}..." if len(summary) > 150 else summary
        
        try:
            # Truncate text if too long
            text_sample = text_content[:1000]
            
            prompt = f"""Summarize the following text in 1-2 concise sentences. Focus on the main topic and key points.
                        Text: {text_sample}
                        Summary:"""
            
            response = self.text_model.invoke(prompt)
            summary = response.content.strip()
            
            # Clean up the summary
            if summary:
                return summary
            else:
                raise ValueError("Empty response from model")
                
        except Exception as e:
            logger.warning(f"ChatGroq text description failed: {e}")
            # Fallback to first 150 characters
            words = text_content.split()[:25]
            return f"{' '.join(words)}..."
            
    def _generate_text_description_old(self, text_content: str) -> str:
        """Generate description for text content using LLM."""
        if not self.text_model or not self.text_tokenizer:
            # Fallback: simple text summarization
            sentences = text_content.split('.')[:2]
            return f"Text content discussing: {' '.join(sentences[:50].split())}..."
        
        try:
            prompt = f"Summarize this text in 2 lines:\n\n{text_content[:500]}"
            inputs = self.text_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.text_model.generate(
                    inputs, 
                    max_length=inputs.shape[1] + 100, 
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.text_tokenizer.eos_token_id
                )
            
            response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception as e:
            logger.warning(f"LLM text description failed: {e}")
            words = text_content.split()[:20]
            return f"Text content about: {' '.join(words)}..."
    
    def _generate_table_description(self, table_chunk: TableChunk) -> str:
        """Generate description for table content."""
        try:
            headers_str = ", ".join(table_chunk.headers) if table_chunk.headers else "No headers"
            description = f"Table with {table_chunk.num_rows} rows and {table_chunk.num_cols} columns. "
            description += f"Column headers: {headers_str}. "
            
            # Add sample data description
            if table_chunk.table_data and len(table_chunk.table_data) > 0:
                sample_row = table_chunk.table_data[0][:3]  # First 3 cells
                description += f"Sample data: {', '.join(map(str, sample_row))}..."
            
            return description
            
        except Exception as e:
            logger.warning(f"Table description generation failed: {e}")
            return f"Table with {len(table_chunk.table_data)} rows and {len(table_chunk.headers)} columns"
    
    def _generate_image_description(self, image_chunk: ImageChunk) -> str:
        """Generate description for image content using Vision-Language Model."""
        if not self.vision_model or not self.vision_processor or not image_chunk.image_base64:
            return f"Image content ({image_chunk.image_format}, {image_chunk.width}x{image_chunk.height})"
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_chunk.image_base64)
            from PIL import Image
            image = Image.open(BytesIO(image_data))
            
            # Generate description using VLM
            inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                output = self.vision_model.generate(**inputs, max_length=50)
            
            description = self.vision_processor.decode(output[0], skip_special_tokens=True)
            return f"Image showing: {description}"
            
        except Exception as e:
            logger.warning(f"VLM image description failed: {e}")
            return f"Image content ({image_chunk.image_format}, {image_chunk.width}x{image_chunk.height})"
    
    def _generate_code_description(self, code_content: str, language: Optional[str]) -> str:
        """Generate description for code content."""
        try:
            lines = code_content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Extract key elements
            functions = [line for line in non_empty_lines if 'def ' in line or 'function' in line]
            classes = [line for line in non_empty_lines if 'class ' in line]
            imports = [line for line in non_empty_lines if 'import ' in line or 'from ' in line]
            
            description = f"Code block in {language or 'unknown language'} with {len(non_empty_lines)} lines. "
            
            if functions:
                description += f"Contains {len(functions)} function(s). "
            if classes:
                description += f"Contains {len(classes)} class(es). "
            if imports:
                description += f"Has {len(imports)} import statement(s). "
                
            # Add first few lines as context
            preview = ' '.join(non_empty_lines[:2])[:100]
            description += f"Preview: {preview}..."
            
            return description
            
        except Exception as e:
            logger.warning(f"Code description generation failed: {e}")
            return f"Code block in {language or 'unknown language'}"


if __name__ == "__main__":
    # Example usage
    parser = DoclingParser()
    
    # Parse a sample document
    try:
        chunks = parser.parse_document("data/SampleReport.pdf")
        
        print(f"Parsed {len(chunks)} chunks:")
        for chunk in chunks:
            print(f"- {chunk.chunk_type.value.title()} Chunk: {chunk.description[:100]}...")

            
    except FileNotFoundError:
        print("Please provide a valid document path for testing.")
    except Exception as e:
        print(f"Error during parsing: {e}")
