# OpenDataLoader PDF — Tutorial Suite

Sample programs covering the major features of
[opendataloader-pdf](https://github.com/opendataloader-project/opendataloader-pdf)
(v2.4.1, Apache 2.0).

## Prerequisites

| Requirement | Notes |
|---|---|
| Java 11+ on PATH | `conda install -n genai -c conda-forge openjdk=11` |
| `opendataloader-pdf` | already in `genai` conda env |
| `langchain-opendataloader-pdf` | already in `genai` conda env (tutorial 10 only) |

Activate the environment before running any tutorial:
```bash
conda activate genai
```

For a fresh environment instead:
```bash
conda env create -f environment.yml
conda activate opendataloader
```

## Test data (`data/`)

| File | Content |
|---|---|
| `sample.pdf` | Generic multi-page document |
| `Systems Thinking in 25 Words or Less.pdf` | Well-structured text document |
| `Systems Thinking Four Key Questions.pdf` | Multi-heading structured document |
| `NVIDIAAn.pdf` | Corporate document with images and URLs |
| `table.pdf` | Document with tabular data |
| `scanned.pdf` | Image-only scanned PDF (needs OCR) |

## Tutorials

| # | File | Feature | Key parameter(s) |
|---|------|---------|-----------------|
| 01 | `tutorial_01_basic_extraction.py` | Single-file Markdown + text | `format="markdown"`, `format="text"` |
| 02 | `tutorial_02_output_formats.py` | All four output formats | `format="markdown,json,html,text"` |
| 03 | `tutorial_03_json_structure.py` | JSON schema analysis — element types, headings, bounding boxes, fonts | `format="json"` |
| 04 | `tutorial_04_table_extraction.py` | Table element isolation | `format="json,markdown"` on `table.pdf` |
| 05 | `tutorial_05_image_extraction.py` | Image output modes | `image_output="off/external/embedded"`, `image_format="png/jpeg"` |
| 06 | `tutorial_06_batch_processing.py` | Whole-folder vs file-list batch | `input_path=[folder]` vs `input_path=[list]` |
| 07 | `tutorial_07_sanitization.py` | PII scrubbing for AI safety | `sanitize=True` |
| 08 | `tutorial_08_struct_tree.py` | Native PDF tags + tagged-PDF export | `use_struct_tree=True`, `format="tagged-pdf"` |
| 09 | `tutorial_09_scanned_ocr.py` | OCR for scanned PDFs (hybrid mode) | `hybrid="docling-fast"` |
| 10 | `tutorial_10_langchain_rag.py` | LangChain RAG pipeline | `OpenDataLoaderPDFLoader`, FAISS |

## Running

```bash
# Run a single tutorial
conda activate genai
cd Code/opendataloader
python tutorial_01_basic_extraction.py

# Run all (non-OCR) tutorials
for i in 01 02 03 04 05 06 07 08 10; do
    python tutorial_0${i}_*.py
done
```

Each tutorial writes its output under `output/<nn>_<name>/`.

## Tutorial 09 — OCR special setup

Tutorial 09 requires the hybrid backend running in a **separate terminal**:

```bash
conda activate genai
opendataloader-pdf-hybrid --port 5002 --force-ocr
```

Then in another terminal:
```bash
conda activate genai
cd Code/opendataloader
python tutorial_09_scanned_ocr.py
```

The script prints graceful instructions if the backend is not reachable.

## Output formats reference

| Format | File ext | Best for |
|--------|----------|---------|
| `markdown` | `.md` | LLM context / RAG ingestion |
| `json` | `.json` | Structured access (bounding boxes, element types) |
| `html` | `.html` | Web display |
| `text` | `.txt` | Plain string processing |
| `tagged-pdf` | `.pdf` | Accessibility-compliant PDF/UA output |
| `annotated-pdf` | `.pdf` | Visual debugging of detected structure |
