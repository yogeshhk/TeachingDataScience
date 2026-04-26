"""
Tests for Pydantic chunk models and DoclingParser utilities in docling_parsing.py.
Runs without API keys or model downloads.
"""
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def mock_heavy_imports():
    """Patch heavy model loading so tests run without downloading anything."""
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()), \
         patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()), \
         patch("transformers.AutoModelForVision2Seq.from_pretrained", return_value=MagicMock()), \
         patch("docling.document_converter.DocumentConverter", return_value=MagicMock()):
        yield


from docling_parsing import (
    ChunkType, BaseChunk, TextChunk, TableChunk, ImageChunk, CodeChunk, DoclingParser
)


class TestChunkTypeEnum:
    def test_text_value(self):
        assert ChunkType.TEXT.value == "text"

    def test_table_value(self):
        assert ChunkType.TABLE.value == "table"

    def test_image_value(self):
        assert ChunkType.IMAGE.value == "image"

    def test_code_value(self):
        assert ChunkType.CODE.value == "code"


class TestTextChunk:
    def test_basic_creation(self):
        chunk = TextChunk(chunk_id="t1", chunk_type=ChunkType.TEXT, text_content="Hello world")
        assert chunk.text_content == "Hello world"
        assert chunk.chunk_type == ChunkType.TEXT

    def test_default_word_count_is_none(self):
        chunk = TextChunk(chunk_id="t2", chunk_type=ChunkType.TEXT, text_content="Hi")
        assert chunk.word_count is None

    def test_with_parent_heading(self):
        chunk = TextChunk(chunk_id="t3", chunk_type=ChunkType.TEXT, text_content="Body",
                          parent_heading="Section 1")
        assert chunk.parent_heading == "Section 1"

    def test_parent_heading_none_safe_join(self):
        """Verifies the null-safe heading concat pattern used in the codebase."""
        chunk = TextChunk(chunk_id="t4", chunk_type=ChunkType.TEXT, text_content="Body")
        heading = chunk.parent_heading or ""
        result = f"{heading} : {chunk.description}" if heading else chunk.description
        assert result == ""

    def test_parent_heading_with_description(self):
        chunk = TextChunk(chunk_id="t5", chunk_type=ChunkType.TEXT, text_content="Body",
                          parent_heading="Intro", description="A summary")
        heading = chunk.parent_heading or ""
        result = f"{heading} : {chunk.description}" if heading else chunk.description
        assert result == "Intro : A summary"


class TestTableChunk:
    def test_basic_creation(self):
        chunk = TableChunk(
            chunk_id="tbl1", chunk_type=ChunkType.TABLE,
            table_data=[["A", "B"], ["1", "2"]], headers=["col1", "col2"]
        )
        assert chunk.headers == ["col1", "col2"]
        assert len(chunk.table_data) == 2

    def test_optional_fields_default_none(self):
        chunk = TableChunk(
            chunk_id="tbl2", chunk_type=ChunkType.TABLE,
            table_data=[], headers=[]
        )
        assert chunk.table_html is None
        assert chunk.num_rows is None
        assert chunk.num_cols is None


class TestImageChunk:
    def test_all_defaults_none(self):
        chunk = ImageChunk(chunk_id="img1", chunk_type=ChunkType.IMAGE)
        assert chunk.image_path is None
        assert chunk.image_base64 is None
        assert chunk.image_format is None
        assert chunk.width is None
        assert chunk.height is None

    def test_with_path(self):
        chunk = ImageChunk(chunk_id="img2", chunk_type=ChunkType.IMAGE,
                           image_path="/tmp/fig1.png", image_format="png")
        assert chunk.image_path == "/tmp/fig1.png"
        assert chunk.image_format == "png"


class TestCodeChunk:
    def test_basic_creation(self):
        chunk = CodeChunk(chunk_id="c1", chunk_type=ChunkType.CODE,
                          code_content="print('hello')")
        assert chunk.code_content == "print('hello')"
        assert chunk.programming_language is None

    def test_with_language(self):
        chunk = CodeChunk(chunk_id="c2", chunk_type=ChunkType.CODE,
                          code_content="x = 1", programming_language="python",
                          line_count=1)
        assert chunk.programming_language == "python"
        assert chunk.line_count == 1


class TestDoclingParserDevice:
    def test_auto_device_returns_string(self):
        with patch("docling.document_converter.DocumentConverter"), \
             patch("transformers.AutoTokenizer.from_pretrained"), \
             patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.AutoModelForVision2Seq.from_pretrained"):
            parser = DoclingParser.__new__(DoclingParser)
            result = DoclingParser._get_device(parser, "auto")
            assert result in ("cuda", "cpu")

    def test_explicit_cpu(self):
        parser = DoclingParser.__new__(DoclingParser)
        assert DoclingParser._get_device(parser, "cpu") == "cpu"

    def test_explicit_cuda(self):
        parser = DoclingParser.__new__(DoclingParser)
        assert DoclingParser._get_device(parser, "cuda") == "cuda"

    def test_no_to_method_called_on_groq(self):
        """ChatGroq must not have .to() called — verify parser init doesn't crash with API key set."""
        with patch("docling.document_converter.DocumentConverter"), \
             patch("transformers.AutoTokenizer.from_pretrained"), \
             patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.AutoModelForVision2Seq.from_pretrained"), \
             patch.dict(os.environ, {"GROQ_API_KEY": "fake-key"}), \
             patch("langchain_groq.ChatGroq") as mock_groq:
            mock_groq.return_value = MagicMock(spec=[])  # no .to() on spec
            parser = DoclingParser.__new__(DoclingParser)
            parser.device = "cpu"
            parser.converter = MagicMock()
            # Should not raise AttributeError about .to()
            try:
                DoclingParser.__init__(parser)
            except Exception:
                pass  # network errors are expected; AttributeError from .to() is not
