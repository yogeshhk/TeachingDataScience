#   Created by Elshad Karimov 
#   Copyright © 2021 AppMillers. All rights reserved.

# Floyd Warshall Algorithm in python

INF = 9999
# Printing the solution
def printSolution(nV, distance):
    for i in range(nV):
        for j in range(nV):
            if(distance[i][j] == INF):
                print("INF", end=" ")
            else:
                print(distance[i][j], end="  ")
        print(" ")


def floydWarshall(nV, G):
    distance = G
    for k in range(nV):
        for i in range(nV):
            for j in range(nV):
                distance[i][j] = min(distance[i][j], distance[i][k]+distance[k][j])
    
    printSolution(nV, distance)

G = [[0, 8, INF,1],
    [INF, 0, 1,INF],
    [4, INF, 0,INF],
    [INF, 2, 9,1]
    ]

floydWarshall(4, G)

