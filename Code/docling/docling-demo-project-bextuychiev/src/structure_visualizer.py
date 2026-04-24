"""
Document structure visualization for Docling processed documents.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from docling_core.types.doc import DoclingDocument


class DocumentStructureVisualizer:
    """Extracts and organizes document structure from Docling documents."""

    def __init__(self, docling_document: DoclingDocument):
        """
        Initialize with a Docling document.

        Args:
            docling_document: The DoclingDocument object from conversion
        """
        self.doc = docling_document

    def get_document_hierarchy(self) -> List[Dict[str, Any]]:
        """
        Extract document hierarchy (headings and structure).

        Returns:
            List of dictionaries containing hierarchical structure info
        """
        hierarchy = []

        if not hasattr(self.doc, 'texts') or not self.doc.texts:
            return hierarchy

        for item in self.doc.texts:
            # Get label to identify item type
            label = getattr(item, 'label', None)

            # Focus on headers and titles
            if label and 'header' in label.lower():
                text = getattr(item, 'text', '')
                prov = getattr(item, 'prov', [])
                page_no = prov[0].page_no if prov else None

                hierarchy.append({
                    'type': label,
                    'text': text,
                    'page': page_no,
                    'level': self._infer_heading_level(label)
                })

        return hierarchy

    def _infer_heading_level(self, label: str) -> int:
        """Infer heading level from label."""
        if 'title' in label.lower():
            return 1
        elif 'section' in label.lower():
            return 2
        elif 'subsection' in label.lower():
            return 3
        else:
            return 4

    def get_tables_info(self) -> List[Dict[str, Any]]:
        """
        Extract table information and convert to DataFrames.

        Returns:
            List of dictionaries with table metadata and DataFrame
        """
        tables_info = []

        if not hasattr(self.doc, 'tables') or not self.doc.tables:
            return tables_info

        for i, table in enumerate(self.doc.tables, 1):
            try:
                # Export table to DataFrame
                df = table.export_to_dataframe(doc=self.doc)

                # Get provenance
                prov = getattr(table, 'prov', [])
                page_no = prov[0].page_no if prov else None

                # Get caption if available
                caption_text = getattr(table, 'caption_text', None)
                caption = caption_text if caption_text and not callable(caption_text) else None

                tables_info.append({
                    'table_number': i,
                    'page': page_no,
                    'caption': caption,
                    'dataframe': df,
                    'shape': df.shape,
                    'is_empty': df.empty
                })

            except Exception as e:
                # Handle tables that can't be converted
                print(f"Warning: Could not process table {i}: {e}")
                continue

        return tables_info

    def get_pictures_info(self) -> List[Dict[str, Any]]:
        """
        Extract picture/image metadata and image data.

        Returns:
            List of dictionaries with picture information and PIL images
        """
        pictures_info = []

        if not hasattr(self.doc, 'pictures') or not self.doc.pictures:
            return pictures_info

        for i, pic in enumerate(self.doc.pictures, 1):
            prov = getattr(pic, 'prov', [])

            if prov:
                page_no = prov[0].page_no
                bbox = prov[0].bbox

                # Get caption if available
                caption_text = getattr(pic, 'caption_text', None)
                caption = caption_text if caption_text and not callable(caption_text) else None

                # Get PIL image if available
                pil_image = None
                try:
                    if hasattr(pic, 'image') and pic.image is not None:
                        if hasattr(pic.image, 'pil_image'):
                            pil_image = pic.image.pil_image
                except Exception as e:
                    print(f"Warning: Could not extract image {i}: {e}")

                pictures_info.append({
                    'picture_number': i,
                    'page': page_no,
                    'caption': caption,
                    'pil_image': pil_image,  # Add PIL image
                    'bounding_box': {
                        'left': bbox.l,
                        'top': bbox.t,
                        'right': bbox.r,
                        'bottom': bbox.b
                    } if bbox else None
                })

        return pictures_info

    def get_document_summary(self) -> Dict[str, Any]:
        """
        Get overall document summary statistics.

        Returns:
            Dictionary with document statistics
        """
        pages = getattr(self.doc, 'pages', {})
        texts = getattr(self.doc, 'texts', [])
        tables = getattr(self.doc, 'tables', [])
        pictures = getattr(self.doc, 'pictures', [])

        # Count different text types
        text_types = {}
        for item in texts:
            label = getattr(item, 'label', 'unknown')
            text_types[label] = text_types.get(label, 0) + 1

        return {
            'name': self.doc.name,
            'num_pages': len(pages) if pages else 0,
            'num_texts': len(texts),
            'num_tables': len(tables),
            'num_pictures': len(pictures),
            'text_types': text_types
        }

    def export_full_structure(self) -> Dict[str, Any]:
        """
        Export complete document structure.

        Returns:
            Dictionary containing all structure information
        """
        return {
            'summary': self.get_document_summary(),
            'hierarchy': self.get_document_hierarchy(),
            'tables': self.get_tables_info(),
            'pictures': self.get_pictures_info()
        }
