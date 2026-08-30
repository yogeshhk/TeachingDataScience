# https://github.com/jerryjliu/llama_index/blob/main/examples/langchain_demo/LangchainDemo.ipynb

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
engine = index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=engine,
        metadata=ToolMetadata(name='Paulindex', description='Provides information about Paul Graham Essay')
    )
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
response = s_engine.query('Explain childhood')
print(response)

### As a chat bot

# tools = [
#     Tool(
#         name="LlamaIndex",
#         func=lambda q: str(index.as_query_engine().query(q)),
#         description="useful for when you want to answer questions about the author. The input to this tool should be a complete english sentence.",
#         return_direct=True
#     ),
# ]

# memory = ConversationBufferMemory(memory_key="chat_history")
# # llm = ChatOpenAI(temperature=0)
# agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=memory)
#
# agent_executor.run(input="hi, i am bob")
