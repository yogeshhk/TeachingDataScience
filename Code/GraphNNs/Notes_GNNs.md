# Graph Neural Networks (GNNs)

## Introduction
Current Machine Learning algorithms like Support Vector Classifier, Naïve Bayes, etc. expect independent features. That’s rarely the case, especially for Natural Language Processing (NLP) applications. Sparse or Dense embeddings used in ML of NLP, are related to each other contextually. Thus, traditional ML algorithms do not learn representation (underlying structure) well and warrant need to inter-dependent representation system such as Graph Neural Networks.

Challenges in NLP would be:
- Using what logic would text paragraph or sentence will become a graph with words as nodes and arcs as ? Including documents as nodes as well? Spectral way of graph formation does not look intuitive
- Using what logic text graph would be convoluted and pooled to form one embedding for whole text sentence/paragraph? Also, should have ability to give node-level ie word embeddings which will incorporate context via neighboring arcs and nodes.

## Potential
- Relatively new and upcoming domain
- Widely applicable: Social Networks, logistics, molecular biology, mesh graphics, etc.
- Active research in Big companies, 
- Usable in MidcurveNN
- Contribution to tf2-gnn via tutorials

## ToDos
-	Build Text Classifier with GNN
- Yao text-gcnn build twitter sentiment corpus files
-	Buy-read Book: https://www.manning.com/books/graph-powered-machine-learning 
-	Deep learning on graphs: successes, challenges, and next steps | Graph Neural Networks https://www.youtube.com/watch?v=PLGcx65MhCc 
-	Graph Node Embedding Algorithms (Stanford) https://www.youtube.com/watch?v=7JELX6DiUxQ 
-	CS224W: Machine Learning with Graphs https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn
-	Spektral: graph deep learning, based on the Keras API and TensorFlow 2  https://graphneural.network/  https://arxiv.org/pdf/2006.12138.pdf 

## Notes
- How to get started with Graph Machine Learning, Aleksa Gordic, https://gordicaleksa.medium.com/how-to-get-started-with-graph-machine-learning-afa53f6f963a
	- Example of Graph based regression: Maybe the nodes in your graphs are tweets and you want to regress the probability that a certain tweet is spreading fake news. Your model would associate a number between 0 and 1 to that node and that would be an example of a node regression task.
	- Graph embedding methods: 
	  - 'Deep Walk': sample “sentences” from the graph by doing random walks.
		- Once you have the sentence you can do the Word2Vec magic and learn your node embeddings such that those nodes that tend to be close in your sentences have similar embeddings.
	- Graph Neural Networks:
		- Spectral methods try to preserve the strict mathematical notion of convolution and had to resort to the frequency domain (Laplacian eigenbasis) for that. Main idea is to project the graph signal into that eigenbasis, filter the projected graph signal directly in the spectral domain by doing an element-wise multiplication with the filter, and reproject it back into the spatial domain. They are computationally expensive, so not used much.
		- Spatial (message passing) methods are not convolutions in the strict mathematical sense of the word, but we still informally call them convolutions, as they were very loosely inspired by the spectral methods. The goal is to update each and every node’s representation. You have the feature vectors from your node and from his neighbors at disposal. GCN uses constants instead of learnable weights as by GAT, while aggregating neighbors.
		  - GraphSAGE — SAmple and aggreGatE, it introduced the concept of sampling your neighborhood and a couple of optimization tricks so as to constrain the compute/memory budget.
		  - PinSage — One of the most famous real-world applications of GNNs — recommender system at Pinterest.

-	Graphs Neural Networks in NLP. Capturing the beautiful semantic… | by Purvanshi Mehta | NeuralSpace | Medium
 ![GNN IO](images/gnnio.png)
 
  -	Although powerful attention mechanisms can automatically learn the syntactic and semantic relationships, it is linear and may have to be constrained to pick correct relationship.
  -	Knowledge Graphs need to be input for NLP applications.


-	A Tutorial on Graph Neural Networks for Natural Language Processing https://shikhar-vashishth.github.io/assets/pdf/emnlp19_tutorial.pdf  https://arxiv.org/abs/1911.03042 

![GNN LSTM](images/gnnlstm.png)

After standard LSTM, in sentiment classification algorithm, we can have Graph relations embedded in GCN and then do classification.

-	Graph Convolutional Networks for Geometric Deep Learning https://towardsdatascience.com/graph-convolutional-networks-for-geometric-deep-learning-1faf17dee008
  - Graph Embedding: transforming a graph to a lower dimension.
  - Graph Convolution: convolutional methods are performed on the input graph itself, with structure and features left unchanged.
  - Images are represented on a 2-dimensional Euclidean grid, where a kernel can move left, right, etc. Graphs are non-Euclidean, and the notion of directions like up, down, etc. don’t have any meaning. Graphs are more abstract, and variables like node degree, proximity, and neighborhood structure provide for more information about the data. So, the ultimate question is: How do we generalize convolutions for graph data?
  - Types:
	  - Spectral Graph Convolutional Networks: Graph Fourier transform. clustered graph would be sparse in the frequency domain allowing for a more efficient representation of the data.
	  - Spatial Graph Convolutional Networks: Neighborhood sampling, Aggregation, Prediction

![GNN Pipe](images/gnnpipe.png)

-	An Introduction to Graph Neural Networks: Models and Applications https://www.youtube.com/watch?v=zCEYiCxrL_0
  - Input is nodes with own vector form. Output of GNNs is again nodes but with context information added to the node vector form. All incident edges info is in built now.
  - This update happens n all nodes, once at each time step, for n epochs.





