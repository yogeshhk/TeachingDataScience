class Node:
    def __init__(self, val):
        self.val = val 
        self.children = []


def bfs(root):
    visited = set() # Hash set for visited nodes
    queue = []

    # Start with the root
    queue.append(root)
    visited.add(root)
    while queue:
        node = queue.pop(0)
        print(node.val)
        for child in node.children:
            if child not in visited:
                queue.append(child)
                visited.add(child)

n1 = Node(7)
n2 = Node(3)
n3 = Node(4)
n4 = Node(5)
n5 = Node(2)
n1.children = [n2, n5]
n2.children = [n3, n4]

bfs(n1)
