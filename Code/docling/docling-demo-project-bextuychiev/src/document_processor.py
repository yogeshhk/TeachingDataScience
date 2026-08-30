"""
Docling integration for processing uploaded documents.
"""

import os
import tempfile
from typing import List, Any
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document processing using Docling."""

    def __init__(self):
        """Initialize the Docling DocumentConverter."""
        # Configure pipeline options for PDF processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True  # Enable image extraction
        pipeline_options.images_scale = 2.0  # Higher resolution for better quality

        # Initialize converter with PDF options
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def process_uploaded_files(self, uploaded_files) -> tuple[List[Document], List[Any]]:
        """
        Process uploaded files and convert them to LangChain Document objects.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Tuple of (LangChain Documents, Docling Documents)
        """
        documents = []
        docling_docs = []
        temp_dir = tempfile.mkdtemp()

        try:
            for uploaded_file in uploaded_files:
                print(f"📄 Processing {uploaded_file.name}...")

                # Save uploaded file to temporary location
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Process the document with Docling
                try:
                    result = self.converter.convert(temp_file_path)

                    # Export to markdown
                    markdown_content = result.document.export_to_markdown()

                    # Create LangChain document
                    doc = Document(
                        page_content=markdown_content,
                        metadata={
                            "filename": uploaded_file.name,
                            "file_type": uploaded_file.type,
                            "source": uploaded_file.name,
                        },
                    )
                    documents.append(doc)

                    # Store the Docling document for structure visualization
                    docling_docs.append({
                        'filename': uploaded_file.name,
                        'doc': result.document
                    })

                    print(f"✅ Successfully processed {uploaded_file.name}")

                except Exception as e:
                    print(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    continue

        finally:
            # Clean up temporary files
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temp directory: {str(e)}")

        print(f"✅ Processed {len(documents)} documents successfully")
        return documents, docling_docs
