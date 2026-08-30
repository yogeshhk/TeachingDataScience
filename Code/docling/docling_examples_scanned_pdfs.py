# https://medium.com/@scholarly360/docling-first-impression-a866a83ac694

from docling.datamodel.base_models import InputFormat
from pathlib import Path

from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrMacOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

if(True):
    input_doc =  Path("./data/scanned.pdf")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Any of the OCR options can be used:EasyOcrOptions, TesseractOcrOptions, TesseractCliOcrOptions, OcrMacOptions(Mac only)
    # ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    # ocr_options = TesseractOcrOptions(force_full_page_ocr=True)
    # ocr_options = OcrMacOptions(force_full_page_ocr=True)
    ocr_options = EasyOcrOptions(force_full_page_ocr=True)
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    doc = converter.convert(input_doc).document
    md = doc.export_to_markdown()
    print(md)