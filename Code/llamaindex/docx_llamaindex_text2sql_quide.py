# https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage,  TextBlock, ImageBlock
from llama_index.core import VectorStoreIndex

import nest_asyncio

nest_asyncio.apply()

llm = Groq(model="llama3-8b-8192") # Need GROQ_API_KEY already set
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5") # HUGGINGFACEHUB_API_TOKEN ?
Settings.llm = llm
Settings.embed_model = embed_model

# use sqlalchemy, a popular SQL database toolkit, create an empty city_stats Table
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, select, insert, text

engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()
# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)
metadata_obj.create_all(engine)

# define SQLDatabase abstraction (a light wrapper around SQLAlchemy).
from llama_index.core import SQLDatabase
sql_database = SQLDatabase(engine, include_tables=["city_stats"])

# add some testing data to SQL database
rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {
        "city_name": "Chicago",
        "population": 2679000,
        "country": "United States",
    },
    {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

# # view current table
# stmt = select(
#     city_stats_table.c.city_name,
#     city_stats_table.c.population,
#     city_stats_table.c.country,
# ).select_from(city_stats_table)

# with engine.connect() as connection:
#     results = connection.execute(stmt).fetchall()
#     print(results)
    
# # Execute a raw SQL query, which directly executes over the table.
# with engine.connect() as con:
#     rows = con.execute(text("SELECT city_name from city_stats"))
#     for row in rows:
#         print(row)


# Use the NLSQLTableQueryEngine to construct natural language queries that are synthesized into SQL queries.
# Need to specify the tables we want to use with this query engine. 
# If we don't the query engine will pull all the schema context, which could overflow the context window of the LLM.

# from llama_index.core.query_engine import NLSQLTableQueryEngine

# query_engine = NLSQLTableQueryEngine(
#     sql_database=sql_database, tables=["city_stats"], llm=llm
# )
# query_str = "Which city has the highest population?"
# response = query_engine.query(query_str)
# print(response)

# If we don't know ahead of time which table we would like to use, and the total size of the table schema overflows your 
# context window size, we should store the table schema in an index so that during query time we can retrieve the right schema.
# The way we can do this is using the SQLTableNodeMapping object, which takes in a SQLDatabase and produces a Node object 
# for each SQLTableSchema object passed into the ObjectIndex constructor.

# from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
# from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
# from llama_index.core import VectorStoreIndex

# table_node_mapping = SQLTableNodeMapping(sql_database)

# # manually set context text
# city_stats_text = (
#     "This table gives information regarding the population and country of a"
#     " given city.\nThe user will query with codewords, where 'foo' corresponds"
#     " to population and 'bar'corresponds to city."
# )

# table_node_mapping = SQLTableNodeMapping(sql_database)
# table_schema_objs = [
#     (SQLTableSchema(table_name="city_stats", context_str=city_stats_text))
# ]

# obj_index = ObjectIndex.from_objects(
#     table_schema_objs,
#     table_node_mapping,
#     VectorStoreIndex,
# )
# query_engine = SQLTableRetrieverQueryEngine(sql_database, obj_index.as_retriever(similarity_top_k=1))
# response = query_engine.query("Which city has the highest population?")
# print(response)
# # you can also fetch the raw result from SQLAlchemy!
# print(response.metadata["result"])

# So far our text-to-SQL capability is packaged in a query engine and consists of both retrieval and synthesis.
# You can use the SQL retriever on its own. 
from llama_index.core.retrievers import NLSQLRetriever

nl_sql_retriever = NLSQLRetriever(
    sql_database, tables=["city_stats"], return_raw=True
)

results = nl_sql_retriever.retrieve(
    "Return the top 5 cities (along with their populations) with the highest population."
)

print(results)

# compose SQL Retriever with standard RetrieverQueryEngine to synthesize a response. The result is roughly similar to packaged Text-to-SQL query engines.
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(nl_sql_retriever)
response = query_engine.query(
    "Return the top 5 cities (along with their populations) with the highest population."
)
print(response.metadata)
print(str(response))