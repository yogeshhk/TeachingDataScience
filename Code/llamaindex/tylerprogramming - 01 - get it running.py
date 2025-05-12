from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()

documents = SimpleDirectoryReader("pdf/").load_data()

index = VectorStoreIndex.from_documents(documents) 

query_engine = index.as_query_engine()

response = query_engine.query("What are the design goals and give details about it please.")

print(response)