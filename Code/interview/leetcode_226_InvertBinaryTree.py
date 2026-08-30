# Invert Binary Tree, meaning Mirror it

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def swap(node):
    if not node:
        return

    # Post Order, LEFT->RIGHT->SELF
    swap(node.left)
    swap(node.right)

    temp = node.left
    node.left = node.right
    node.right = temp


def invert_tree(root):
    swap(root)
    return root

# Driver function to test above function
root = Node(10)
root.left = Node(5)
root.right = Node(50)
root.left.left = Node(1)
root.right.right = Node(100)
root.right.left = Node(40)
invert_tree(root)
print("Root ", root.data, root.left.data, root.right.data)