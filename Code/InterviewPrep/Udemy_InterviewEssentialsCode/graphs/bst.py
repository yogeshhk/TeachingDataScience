class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def insert(root, node):
    if root == None:
        root = node
        return

    if (root.val < node.val):
        # Insert right
        if root.right == None:
            root.right = node
        else:
            insert(root.right, node)
    else:
        # Insert left
        if root.left == None:
            root.left = node
        else:
            insert(root.left, node)

def search(root, val):
    if root == None or root.val == val:
        return root
    if root.val < val:
        return search(root.right, val)
    else:
        return search(root.left, val)

def print_tree(root):
    if root == None:
        return
    print_tree(root.left)
    print(root.val)
    print_tree(root.right)


root = Node(5)
insert(root, Node(2))
insert(root, Node(6))
insert(root, Node(1))
insert(root, Node(10))
print_tree(root)

print(search(root, 5))
print(search(root, 2))
print(search(root, 6))
print(search(root, 1))
print(search(root, 10))
print(search(root, 3))