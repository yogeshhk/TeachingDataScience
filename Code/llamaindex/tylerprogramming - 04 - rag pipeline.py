import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

load_dotenv()

# 1. Load data
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# 2. Create index
# from llama_index.core import VectorStoreIndex

# index = VectorStoreIndex.from_documents(documents)

# print(documents)

# 3. Create vector store index with Chroma
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context
# )

# load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)

# 4. Create query engine
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)

# query
response = query_engine.query("What is the meaning of life?")
print("***********")
print(response)