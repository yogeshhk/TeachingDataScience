from __future__ import annotations
from typing import List
from decorators import empty, remove


class Node:
    def __init__(self, parent: Node, values: List[int]):
        self.values = values
        self.children = []
        self.parent = parent

    @remove
    def add(self, val: int):
        # We only add a value to an leaf node.
        # Those have no children, so simply append and sort values.
        self.values.append(val)
        self.values.sort()

    @empty
    # Splits the current node
    def split(self):
        val = self.values.pop(1)  # This value moves to the parent
        parent = self.parent

        # Find index to add val in parent's values
        if parent:
            index = len(parent.values)
            for i in range(len(parent.values)):
                if val <= parent.values[i]:
                    index = i
                    break  # Correct location found
            parent.values.insert(index, val)
        else:
            # If this is the root, create a new parent
            self.parent = Node(None, [val])
            self.parent.children.append(self)
            parent = self.parent
            index = 0

        # Make a new left sibling node
        # Note that you could also choose to make a new right sibling node
        left = Node(parent, [self.values.pop(0)])

        if self.children:
            left.children = self.children[:2]
            for child in self.children[:2]:
                child.parent = left
            self.children = self.children[2:]

        # Attach sibling node to parent
        parent.children.insert(index, left)
        assert(len(self.values) == 1)

    # Returns True iff the current node is a leaf node
    def is_leaf(self):
        return len(self.children) == 0

    # Returns a string representing this node
    def __str__(self):
        return f"<Node values [{' '.join(map(str, self.values)) if self.values else ''}] children " \
            f"[{', '.join([str(node) for node in self.children]) if self.children else ''}]>"


class Tree:
    @empty
    def __init__(self, root: Node = None):
        self.root = root

    @empty
    # Adds the given value to the tree
    def add(self, val: int):
        if self.root is None:
            self.root = Node(None, [val])
        else:
            self.get_leaf(val).add(val)

    @empty
    # Returns the leaf node the given value should be added to
    def get_leaf(self, val: int) -> Node:
        node = self.root
        # Start from the root, work down until we find the leaf node
        while node is not None:
            if len(node.values) == 3:
                # If we find a full node along the way, split it
                node.split()
                node = node.parent
                if node.parent is None:
                    # If the split node was the root, make the parent the new root
                    self.root = node
            if node.is_leaf():
                # This is where the value needs to go
                return node
            else:
                # Find the child we need to search in
                node_index = len(node.values)
                for i in range(len(node.values)):
                    if val <= node.values[i]:
                        node_index = i
                        break
                node = node.children[node_index]  # Continue searching from this node

    @empty
    # Returns true iff the given value is in the tree
    def contains(self, val: int) -> bool:
        node = self.root
        while node is not None:
            if val in node.values:
                # Found it
                return True
            elif not node.is_leaf():
                # Search in the child node that can contain this value
                index = len(node.values)
                for i in range(len(node.values)):
                    if val <= node.values[i]:
                        index = i
                        break
                node = node.children[index]
            else:
                # Not in the tree
                return False

    # Returns a string representing this tree
    def __str__(self):
        return f"<Tree [{self.root if self.root else ''}]>"
