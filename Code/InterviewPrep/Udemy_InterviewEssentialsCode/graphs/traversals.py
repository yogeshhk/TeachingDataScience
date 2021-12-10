class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder(root):
    if root is not None:
        inorder(root.left)
        print(root.val)
        inorder(root.right)

def preorder(root):
    if root is not None:
        print(root.val)
        preorder(root.left)
        preorder(root.right)

def postorder(root):
    if root is not None:
        postorder(root.left)
        postorder(root.right)
        print(root.val)

root = Node(6)
root.right = Node(7)
root.left = Node(3)
root.left.left = Node(1)
root.left.right = Node(5)

print("Inorder:")
inorder(root)
print("Preorder:")
preorder(root)
print("Postorder:")
postorder(root)