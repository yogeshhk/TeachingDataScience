# Graph Neural Networks (GNNs)

Introduction
Current Machine Learning algorithms like Support Vector Classifier, Naïve Bayes, etc. expect independent features. That’s rarely the case, especially for Natural Language Processing (NLP) applications. Sparse or Dense embeddings used in ML of NLP, are related to each other contextually. Thus, traditional ML algorithms do not learn representation (underlying structure) well and warrant need to inter-dependent representation system such as Graph Neural Networks.
To-dos
-	Build Clause Classifier with GNN
Notes
-	Graphs Neural Networks in NLP. Capturing the beautiful semantic… | by Purvanshi Mehta | NeuralSpace | Medium
 ![GNN IO](images/gnnio.png)
 
o	Although powerful attention mechanisms can automatically learn the syntactic and semantic relationships, it is linear and may have to be constrained to pick correct relationship.
o	Knowledge Graphs need to be input for NLP applications.

-	Graph Node Embedding Algorithms (Stanford) https://www.youtube.com/watch?v=7JELX6DiUxQ 
-	CS224W: Machine Learning with Graphs https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn
-	Spektral: graph deep learning, based on the Keras API and TensorFlow 2  https://graphneural.network/  https://arxiv.org/pdf/2006.12138.pdf 
-	A Tutorial on Graph Neural Networks for Natural Language Processing https://shikhar-vashishth.github.io/assets/pdf/emnlp19_tutorial.pdf  https://arxiv.org/abs/1911.03042 

![GNN LSTM](images/gnnlstm.png)

After standard LSTM, in sentiment classification algorithm, we can have Graph relations embedded in GCN and then do classification.
-	Book: https://www.manning.com/books/graph-powered-machine-learning 
-	Using Graph Convolutional Neural Networks for NLP tasks http://cs.virginia.edu/~ms5sw/GCN_for_NLP.pdf 
-	Deep learning on graphs: successes, challenges, and next steps | Graph Neural Networks https://www.youtube.com/watch?v=PLGcx65MhCc 
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





