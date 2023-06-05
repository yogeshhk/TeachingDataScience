class Tree:

    def height(self) -> int:
        return 1 + self.maxdepth(self.root)

    def maxdepth(self, node) -> int:
        if len(node.children()) = 0:
            return 0  # We found a leaf
        else:
            return 1 + max(self.maxdepth(child) for child in node.children)
