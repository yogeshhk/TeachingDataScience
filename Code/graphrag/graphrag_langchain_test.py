# https://stackoverflow.com/questions/78845567/llm-unable-to-reliably-retrieve-nodes-or-information-from-a-knowledge-graph-usin

import networkx as nx
from langchain_community.graphs.index_creator import GraphIndexCreator
from langchain_openai import ChatOpenAI
import os

groq_api_key = os.environ.get("GROQ_API_KEY")
# api_key='<_groq_api_here> #https://console.groq.com/keys'

llm = ChatOpenAI(
    openai_api_base="https://api.groq.com/openai/v1", 
    openai_api_key=groq_api_key,
    model_name="llama3-70b-8192", #https://console.groq.com/docs/models
)

index_creator = GraphIndexCreator(llm=llm)

#Reading nodes and edges from text
docs1 = [
'Joanie loves Chachi',
'Johny is here',
'I still love Jennay'
]

text1 = '\n'.join(docs1)

# Create a knowledge graph
graph = nx.Graph()
graph = index_creator.from_text(text1)

# created graph inspection
print('Triples output:',graph.get_triples())

# Graph can be saved and loaded later on
"""
print('saving the graph ->','graph1.gml')
graph.write_to_gml("graph1.gml")
"""

# if it was saved, you do not have to build the graph each single time, you can load a graph previously built 
"""
from langchain_community.graphs import NetworkxEntityGraph
graph = NetworkxEntityGraph.from_gml("graph1.gml")
print(graph.get_triples())
print(graph.get_number_of_nodes())
"""

#Retrieval
from langchain.chains import GraphQAChain
chain = GraphQAChain.from_llm(llm, graph=graph, verbose=True)

query1='Is Joanie part of the input data?'
query2="Tell me about Joanie's relation to the other people?"
print(chain.invoke(query1))
print(chain.invoke(query2))

#Optional drawing - using pyvis
"""
from pyvis.network import Network

# Assuming `graph` is the Networkx graph you created with GraphIndexCreator
G = graph._graph

# Convert to PyVis graph
net = Network(notebook=True)
net.from_nx(G)

# Display the graph
net.show("graph1.html")
"""