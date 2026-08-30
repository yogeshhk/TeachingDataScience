# https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html
# Using LlamaIndex as a Callable Tool

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

documents = SimpleDirectoryReader('data/experiment').load_data()
repo_id = "tiiuae/falcon-7b"

Settings.llm = HuggingFaceInferenceAPI(model_name=repo_id, temperature=0.1, max_new_tokens=1024)
Settings.embed_model = HuggingFaceEmbedding()

index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine(similarity_top_k=3)

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(name='pg_essay', description='Paul Graham essay on What I Worked On')
    )
]

query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
# response = s_engine.query('Explain childhood')
response = query_engine.query('How was Paul Grahams life different before and after YC?')

print(response)
