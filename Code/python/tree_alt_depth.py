class Tree:

    def depth(self) -> int:
        if len(self.parent) is None:
            return 0
        return 1 + self.parent.depth()
