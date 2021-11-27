class Tree:

    def depth(self, node) -> int:
        if node == self.root:
            return 0
        else:
            return 1 + self.depth(node.parent)
