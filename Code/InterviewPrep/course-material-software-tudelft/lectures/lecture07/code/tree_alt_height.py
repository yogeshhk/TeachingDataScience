class Tree:

    def height(self) -> int:
        if len(self.children) == 0:
            return 1
        return 1 + max(c.height() for c in self.children)
