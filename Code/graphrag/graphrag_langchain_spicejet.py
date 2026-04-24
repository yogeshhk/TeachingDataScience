# https://medium.com/data-science-in-your-pocket/graph-analytics-relationship-link-prediction-in-graphs-using-neo4j-79a81716e73a
# https://www.youtube.com/watch?v=3B6VjDtbsbw&t=2s

import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.chains import graph_qa
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain.indexes import GraphIndexCreator

groq_api_key = os.environ.get("GROQ_API_KEY")
# api_key='<_groq_api_here> #https://console.groq.com/keys'

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1", 
    openai_api_key=groq_api_key,
    model_name="llama3-70b-8192", #https://console.groq.com/docs/models
)


df = pd.read_csv("data/flight_network_spicejet.csv")
print(df.head())

def distance(x):
    if x < 500:
            return "near, short distance"
    elif x < 1000:
          return " double distance but not very close"
    else:
          return 'far away'

df['distance'] = df.apply(lambda x: distance(x['distance']),axis=1)

graph = NetworkxEntityGraph()

# Add nodes to the graph
for id, row in df.iterrows():
      graph.add_node(row['origin'])
      graph.add_node(row['destination'])

# Add Edges to the graph
for id, row in df.iterrows():
      graph._graph.add_edge(row['origin'],
                            row['destination'],
                            relation=row['distance'])

chain = GraphQAChain.from_llm(
    llm=llm, 
    graph=graph, 
    verbose=True
)

question = """Places to visit near Hyderabad?"""
chain.run(question)

question = """Is Mumbai far away from Varanasi?"""
chain.run(question)