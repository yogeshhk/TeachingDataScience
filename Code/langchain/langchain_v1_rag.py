import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

# 1. Load Local PDF
# Replace "your_document.pdf" with your actual file path
loader = PyPDFLoader("data/ag-studio.pdf")
docs = loader.load()

# 2. Split Documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = splitter.split_documents(docs)

# 3. Embeddings & Vector Store (Updated to langchain-chroma)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=embeddings,
    collection_name="local-pdf-rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context:
{context}

Question: {question}
""")

# 5. Initialize Model (v1.0 way)
model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0
)

# 6. Construct the LCEL Chain
# Note: retriever | format_docs ensures the model gets text, not Document objects
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt 
    | model 
    | StrOutputParser()
)

# 7. Execution
response = chain.invoke("What is the main topic of this document?")
print(response)