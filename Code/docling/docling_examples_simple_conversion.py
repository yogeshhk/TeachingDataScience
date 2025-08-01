# https://docling-project.github.io/docling/examples/minimal/

from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL

convertor = DocumentConverter()
doc = convertor.convert(source=source).document

print(doc.export_to_markdown())
# output: ## Docling Technical Report [...]"