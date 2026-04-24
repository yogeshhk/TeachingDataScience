# https://docs.llamaindex.ai/en/stable/examples/cookbooks/llama3_cookbook_groq/

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage
from llama_index.core import VectorStoreIndex

import nest_asyncio

nest_asyncio.apply()

llm = Groq(model="llama3-8b-8192")
llm_70b = Groq(model="llama3-70b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

# response = llm.complete("do you like drake or kendrick better?")

# print(response)

# stream_response = llm.stream_complete(
#     "you're a drake fan. tell me why you like drake more than kendrick"
# )

# for t in stream_response:
#     print(t.delta, end="")

# messages = [
#     ChatMessage(role="system", content="You are Kendrick."),
#     ChatMessage(role="user", content="Write a verse."),
# ]
# response = llm.chat(messages)
# print(response)

# docs_kendrick = LlamaParse(result_type="text").load_data("./data/cookbook/kendrick.pdf")
# docs_drake = LlamaParse(result_type="text").load_data("./data/cookbook/drake.pdf")
# docs_both = LlamaParse(result_type="text").load_data("./data/cookbook/drake_kendrick_beef.pdf")

# docs_kendrick = SimpleDirectoryReader(input_files=["data/cookbook/kendrick.pdf"]).load_data()
# docs_drake = SimpleDirectoryReader(input_files=["data/cookbook/drake.pdf"]).load_data()
# docs_both = SimpleDirectoryReader(input_files=["data/cookbook/drake_kendrick_beef.pdf"]).load_data()

# index = VectorStoreIndex.from_documents(docs_both)
# query_engine = index.as_query_engine(similarity_top_k=3)
# response = query_engine.query("Tell me about family matters")
# print(str(response))

# from llama_index.core import SummaryIndex

# summary_index = SummaryIndex.from_documents(docs_both)
# summary_engine = summary_index.as_query_engine()
# response = summary_engine.query(
#     "Given your assessment of this article, who won the beef?"
# )
# print(str(response))

# from llama_index.core.tools import QueryEngineTool, ToolMetadata

# vector_tool = QueryEngineTool(
#     index.as_query_engine(),
#     metadata=ToolMetadata(
#         name="vector_search",
#         description="Useful for searching for specific facts.",
#     ),
# )

# summary_tool = QueryEngineTool(
#     index.as_query_engine(response_mode="tree_summarize"),
#     metadata=ToolMetadata(
#         name="summary",
#         description="Useful for summarizing an entire document.",
#     ),
# )

# from llama_index.core.query_engine import RouterQueryEngine

# query_engine = RouterQueryEngine.from_defaults(
#     [vector_tool, summary_tool], select_multi=False, verbose=True, llm=llm_70b
# )

# response = query_engine.query(
#     "Tell me about the song meet the grahams - why is it significant"
# )

# from sqlalchemy import (
#     create_engine,
#     MetaData,
#     Table,
#     Column,
#     String,
#     Integer,
#     select,
#     column,
# )

# engine = create_engine("sqlite:///data/cookbook/chinook/chinook.db")
# from llama_index.core import SQLDatabase

# sql_database = SQLDatabase(engine)

# from llama_index.core.indices.struct_store import NLSQLTableQueryEngine

# query_engine = NLSQLTableQueryEngine(
#     sql_database=sql_database,
#     tables=["albums", "tracks", "artists"],
#     llm=llm_70b,
# )

# response = query_engine.query("What are some albums?")

# print(response)

# response = query_engine.query("What are some artists? Limit it to 5.")

# print(response)

# response = query_engine.query(
#     "What are some tracks from the artist AC/DC? Limit it to 3"
# )

# print(response)

# print(response.metadata["sql_query"])

# from llama_index.llms.groq import Groq
# from llama_index.core.prompts import PromptTemplate
# from pydantic import BaseModel


# class Restaurant(BaseModel):
#     """A restaurant with name, city, and cuisine."""

#     name: str
#     city: str
#     cuisine: str


# llm = Groq(model="llama3-8b-8192", pydantic_program_mode="llm")
# prompt_tmpl = PromptTemplate(
#     "Generate a restaurant in a given city {city_name}"
# )

# restaurant_obj = llm.structured_predict(
#     Restaurant, prompt_tmpl, city_name="Mumbai"
# )
# print(restaurant_obj)

import json
from typing import Sequence, List

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b


multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm_70b,
    verbose=True,
)

response = agent.chat("What is (121 + 2) * 5?")
print(str(response))