import opensource_settings

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("pdf/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

query = "What are the design goals and give details about it please."

response = query_engine.query(query)

print(response)
