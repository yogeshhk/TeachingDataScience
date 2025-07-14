from typing import Iterator
import time
import os
import warnings

# Suppress warnings and set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

# Fix for bottleneck compatibility issues
import numpy as np
if hasattr(np, '_NoValue'):
    np._NoValue = np._NoValue
else:
    np._NoValue = object()

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

    def get_docling_document(self, file_path: str = None):
        """Returns the raw docling document for display purposes"""
        if file_path is None:
            file_path = self.file_path
        
        docling_doc = self.converter.convert(file_path).document
        return docling_doc


class BookQASystem:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.chat_history = []
        
    def initialize_embeddings(self):
        """Initialize the embedding model"""
        if self.embeddings is None:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return self.embeddings
    
    def create_vectorstore(self, pdf_path: str, progress_callback=None):
        """Create or load vector store from PDF"""
        if progress_callback:
            progress_callback("Initializing embedding model...")
        
        self.initialize_embeddings()
        
        index_path = f"{pdf_path}_faiss_index"
        
        if os.path.exists(index_path):
            if progress_callback:
                progress_callback("Loading existing vector store...")
            
            self.vectorstore = FAISS.load_local(
                index_path, self.embeddings, allow_dangerous_deserialization=True
            )
        else:
            if progress_callback:
                progress_callback("Creating new vector store...")
            
            loader = DoclingBookLoader(pdf_path)
            documents = loader.load()
            
            if progress_callback:
                progress_callback("Splitting document into chunks...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            
            if progress_callback:
                progress_callback("Building vector store and creating embeddings...")
            
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            
            if progress_callback:
                progress_callback("Saving vector store...")
            
            self.vectorstore.save_local(index_path)
        
        return self.vectorstore
    
    def create_qa_chain(self, book_name: str):
        """Create the QA chain"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
        
        llm = ChatOpenAI(
            model="local-model",
            openai_api_base="http://localhost:1234/v1",
            openai_api_key="not-needed",
            temperature=0,
        )
        
        template = """You are a helpful assistant answering questions about the book: {book_name}. 
        
        Use the following context to answer the question: {context}
        
        Question: {question}
        
        Answer the question accurately and concisely based on the context provided."""
        
        prompt = PromptTemplate(
            input_variables=["book_name", "context", "question"], template=template
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_variable_name": "context",
            },
        )
        
        return self.qa_chain
    
    def ask_question(self, question: str, book_name: str):
        """Ask a question to the QA system"""
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")
        
        result = self.qa_chain.invoke({
            "question": question,
            "chat_history": self.chat_history,
            "book_name": book_name,
        })
        
        self.chat_history.append((question, result["answer"]))
        return result
    
    def reset_chat_history(self):
        """Reset the chat history"""
        self.chat_history = []
