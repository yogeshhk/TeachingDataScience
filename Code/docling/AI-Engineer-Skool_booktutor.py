# https://github.com/AI-Engineer-Skool/booktutor-ai/blob/main/booktutor.py
from typing import Iterator
import time
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.documents import Document as LCDocument
from langchain_core.document_loaders import BaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.prompts import PromptTemplate


class DoclingBookLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.AUTO
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        print(f"\n📚 Processing book: {self.file_path}")

        process_start = time.time()
        docling_doc = self.converter.convert(self.file_path).document
        process_time = time.time() - process_start
        print(f"✅ Book processed successfully in {process_time:.2f} seconds")

        print("🔄 Converting to markdown format...")
        convert_start = time.time()
        text = docling_doc.export_to_markdown()
        convert_time = time.time() - convert_start
        print(f"✅ Conversion complete in {convert_time:.2f} seconds")

        metadata = {
            "source": self.file_path,
            "format": "book",
            "process_time": process_time,
            "convert_time": convert_time,
        }

        yield LCDocument(page_content=text, metadata=metadata)


def create_book_qa_system(pdf_path: str):
    total_start_time = time.time()
    print("\n🚀 Initializing Book QA System...")

    index_path = f"{pdf_path}_faiss_index"

    print("🔤 Initializing embedding model...")
    embedding_start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_init_time = time.time() - embedding_start
    print(f"✅ Embedding model initialized in {embedding_init_time:.2f} seconds")

    if os.path.exists(index_path):
        print("📦 Loading existing vector store...")
        load_start = time.time()
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        load_time = time.time() - load_start
        print(f"✅ Vector store loaded in {load_time:.2f} seconds")
    else:
        print("\n💫 No existing index found. Creating new one...")

        loader = DoclingBookLoader(pdf_path)
        documents = loader.load()

        print("\n📄 Splitting document into chunks...")
        split_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        split_time = time.time() - split_start
        print(f"✅ Created {len(splits)} chunks in {split_time:.2f} seconds")

        print("\n📦 Building vector store and creating embeddings...")
        vectorstore_start = time.time()
        vectorstore = FAISS.from_documents(splits, embeddings)
        vectorstore_time = time.time() - vectorstore_start
        print(f"✅ Vector store built in {vectorstore_time:.2f} seconds")

        print(f"💾 Saving vector store to {index_path}")
        save_start = time.time()
        vectorstore.save_local(index_path)
        save_time = time.time() - save_start
        print(f"✅ Vector store saved in {save_time:.2f} seconds")

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    print("✅ Vector store ready")

    print("\n🤖 Connecting to local language model...")
    llm = ChatOpenAI(
        model="local-model",
        openai_api_base="http://localhost:1234/v1",
        openai_api_key="not-needed",
        temperature=0,
    )

    print("⛓️ Creating QA chain...")

    template = """You are a helpful assistant answering questions about the book: {book_name}. 
    
    Use the following context to answer the question: {context}
    
    Question: {question}
    
    Answer the question accurately and concisely based on the context provided."""

    prompt = PromptTemplate(
        input_variables=["book_name", "context", "question"], template=template
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
    )

    total_time = time.time() - total_start_time
    print(f"\n✨ System ready! Total setup took {total_time:.2f} seconds")

    return qa_chain


def print_result(result):
    print("\n" + "=" * 80)
    print("📚 RETRIEVED CONTEXT CHUNKS:")
    print("=" * 80)

    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\nCHUNK {i}:")
        print("-" * 40)
        print(doc.page_content)
        print("-" * 40)

    print("\n" + "=" * 80)
    print("🤖 LLM RESPONSE:")
    print("=" * 80 + "\n")
    print(result["answer"])
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Interactive QA system for PDF books")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File '{args.pdf_path}' not found")
        return

    qa_system = create_book_qa_system(args.pdf_path)
    chat_history = []

    print("\n📚 Ready to answer questions about your PDF!")
    print("Type 'quit' to exit")

    while True:
        question = input("\n❓ Ask a question: ")
        if question.lower() == "quit":
            break

        print("\n🔄 Processing your question...")
        result = qa_system.invoke(
            {
                "question": question,
                "chat_history": chat_history,
                "book_name": os.path.basename(args.pdf_path),
            }
        )

        print_result(result)
        chat_history.append((question, result["answer"]))


if __name__ == "__main__":
    main()
