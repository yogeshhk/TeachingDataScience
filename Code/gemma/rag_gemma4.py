from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# 1. Data
sample_data = [
    "Gemma 3 1B is a lightweight model by Google, ideal for devices with low RAM.",
    "Ollama allows running large language models locally on your CPU or GPU.",
    "RAG stands for Retrieval-Augmented Generation, which helps LLMs use private data.",
    "ChromaDB is a popular open-source vector database for storing document embeddings.",
    "4 GB of RAM is the absolute minimum for running local RAG with 1B models."
]

documents = [Document(page_content=text) for text in sample_data]

# 2. Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(documents)

# 3. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    model_kwargs={"trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True}
)

# 4. Vector store
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 5. Gemma 4 via LM Studio
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="google/gemma-3-4b", # "gemma-4-e2b-it",
    temperature=0
)

# 6. Prompt
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.

Context:
{context}

Question:
{question}
"""
)

# 7. RAG chain (modern LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Query
question = "Which model is best for low RAM devices?"
response = rag_chain.invoke(question)

print(response)