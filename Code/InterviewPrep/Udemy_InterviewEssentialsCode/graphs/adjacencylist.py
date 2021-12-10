class Node:
    def __init__(self, val):
        self.val = val 
        self.children = []

n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n1.children.append(n2)
n1.children.append(n3)

for child in n1.children:
    print(child.val) # 2, 3