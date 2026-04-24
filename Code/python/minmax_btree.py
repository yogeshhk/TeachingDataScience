class BinarySearchTree:

    def min(self) -> int:
        if self.left is not None:
            return self.left.min()
        return self.value
