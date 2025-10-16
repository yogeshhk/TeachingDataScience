# https://www.geeksforgeeks.org/count-bst-nodes-that-are-in-a-given-range/
# Given a Binary Search Tree (BST) and a range, count number of nodes that lie in the given range.
# Examples:
#
#
# Input:
#         10
#       /    \
#     5       50
#    /       /  \
#  1       40   100
# Range: [5, 45]
#
# Output:  3
# There are three nodes in range, 5, 10 and 40

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def count_BST_nodes(root, myrange, count):
    print("Visiting ", root.data)
    if myrange[0] <= root.data <= myrange[1]:
        count += 1

    if root.left is None and root.right is None:
        return count

    if root.left is not None:
        count = count_BST_nodes(root.left, myrange, count)
    if root.right is not None:
        count = count_BST_nodes(root.right, myrange, count)

    return count


# Driver function to test above function
root = Node(10)
root.left = Node(5)
root.right = Node(50)
root.left.left = Node(1)
root.right.right = Node(100)
root.right.left = Node(40)
myrange = [5, 45]
count = 0
print("Count ", count_BST_nodes(root, myrange, count))
