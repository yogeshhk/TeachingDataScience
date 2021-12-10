class Node:
    def __init__(self, val):
        self.val = val 
        self.children = []


visited = set() # Hash set for visited nodes
def dfs(visited, node):
    if node not in visited:
        print(node.val)
        visited.add(node)
        for child in node.children:
            dfs(visited, child)

n1 = Node(7)
n2 = Node(3)
n3 = Node(4)
n4 = Node(5)
n5 = Node(2)
n1.children = [n2, n5]
n2.children = [n3, n4]

dfs(visited, n1)