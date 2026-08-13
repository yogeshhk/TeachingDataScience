# Find max depth of binary tree
# Example [3,9,20,null,null,15,7] => 3

# Definition of singly-linked list
class ListNode:
    def __init__(self, val=0, left=None,right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root):
    if not root:
        return 0
    return (1 + max(maxDepth(root.left),maxDepth(root.right)))
