class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def max_depth(root):
    if root == None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

'''
Create the following tree:
        1
    2        5
 3     4
'''
n1 = Node(1)
n2 = Node(2)
n3 = Node(3)
n4 = Node(4)
n5 = Node(5)
n1.left = n2
n1.right = n5
n2.left = n3
n2.right = n4

print(max_depth(n1)) # 3
print(max_depth(n2)) # 2
print(max_depth(n3)) # 1