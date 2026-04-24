# https://gpt-index.readthedocs.io/en/latest/examples/query_engine/sub_question_query_engine.html
# Using LlamaIndex as a Callable Tool

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain import HuggingFaceHub
from llama_index import LangchainEmbedding, ServiceContext

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.query_engine import SubQuestionQueryEngine

documents = SimpleDirectoryReader('data/experiment').load_data()
repo_id = "tiiuae/falcon-7b"
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, 'truncation': 'only_first',
                                                    "max_length": 1024})
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(chunk_size=512, llm_predictor=llm_predictor, embed_model=embed_model)

index = VectorStoreIndex.from_documents(documents=documents, service_context=service_context)
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
