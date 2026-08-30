import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

parser = LlamaParse(
    api_key="llx-FEEOy75qNytHDasHebdwMBpf1HyPc1lPzs2qdyru251Yx98T",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(
    "./pdf", file_extractor=file_extractor
).load_data()

index = VectorStoreIndex.from_documents(documents) 
index.storage_context.persist(persist_dir="storage")

query_engine = index.as_query_engine()

response = query_engine.query("What are the design goals and give details about it please.")

print(response)


