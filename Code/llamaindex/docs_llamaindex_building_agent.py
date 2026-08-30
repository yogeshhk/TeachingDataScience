# https://docs.llamaindex.ai/en/stable/understanding/agent/

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage,  TextBlock, ImageBlock
from llama_index.core import VectorStoreIndex
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.agent.workflow import FunctionAgent

import nest_asyncio

nest_asyncio.apply()

llm = Groq(model="llama3-8b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="You are an agent that can perform basic mathematical operations using tools.",
)

async def main():
    response = await workflow.run(user_msg="What is 20+(2*4)?")
    print(response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())