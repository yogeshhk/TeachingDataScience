#   Created by Elshad Karimov 
#   Copyright © 2021 AppMillers. All rights reserved.

# Prims Algorithm  in Python
import sys
class Graph:
    def __init__(self, vertexNum, edges, nodes):
        self.edges = edges
        self.nodes = nodes
        self.vertexNum = vertexNum
        self.MST = []
    
    def printSolution(self):
        print("Edge : Weight")
        for s, d, w in self.MST:
            print("%s -> %s: %s" % (s, d, w))
    
    def primsAlgo(self):
        visited = [0]*self.vertexNum
        edgeNum=0
        visited[0]=True
        while edgeNum<self.vertexNum-1:
            min = sys.maxsize
            for i in range(self.vertexNum):
                if visited[i]:
                    for j in range(self.vertexNum):
                        if ((not visited[j]) and self.edges[i][j]):
                            if min > self.edges[i][j]:
                                min = self.edges[i][j]
                                s = i
                                d = j
            self.MST.append([self.nodes[s], self.nodes[d], self.edges[s][d]])
            visited[d] = True
            edgeNum += 1
        self.printSolution()



edges = [[0, 10, 20, 0, 0],
		[10, 0, 30, 5, 0],
		[20, 30, 0, 15, 6],
		[0, 5, 15, 0, 8],
		[0, 0, 6, 8, 0]]
nodes = ["A","B","C","D","E"]
g = Graph(5, edges, nodes)
g.primsAlgo()
