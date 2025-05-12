import os
from dotenv import load_dotenv

import llama_index.core

load_dotenv()

# general usage 
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

documents = SimpleDirectoryReader("pdf/").load_data()


if os.path.exists("storage"):
    print("Loading index from storage")
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    index = load_index_from_storage(storage_context)
else:
    print("Creating new index")
    index = VectorStoreIndex.from_documents(documents) 
    index.storage_context.persist(persist_dir="storage")

query_engine = index.as_query_engine()

response = query_engine.query("What are the design goals and give details about it please.")
print(response)
