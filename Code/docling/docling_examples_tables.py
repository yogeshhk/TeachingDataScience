# https://medium.com/@scholarly360/docling-first-impression-a866a83ac694

from docling.document_converter import DocumentConverter
from pprint import pprint
from pathlib import Path
import pandas as pd

source = "./data/sample.pdf"   
converter = DocumentConverter()
result = converter.convert(source)
# to markdown
pprint(result.document.export_to_markdown())

# to text
pprint(result.document.export_to_text())

if(True):
    input_doc_path = Path("./data/table.pdf")
    output_dir = Path("scratch")

    doc_converter = DocumentConverter()
    conv_res = doc_converter.convert(input_doc_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_res.input.file.stem

    # Export tables
    for table_ix, table in enumerate(conv_res.document.tables):
        table_df: pd.DataFrame = table.export_to_dataframe()
        print(f"## Table {table_ix}")
        print(table_df.to_markdown())

        # Save the table as csv
        element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix+1}.csv"
        table_df.to_csv(element_csv_filename)

        # Save the table as html
        element_html_filename = output_dir / f"{doc_filename}-table-{table_ix+1}.html"
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html())

### At the end CSV and HTML will be genertated
### With printing on console also