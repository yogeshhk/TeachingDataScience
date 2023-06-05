class BinarySearchTree:

    def contains(self, val: int) -> bool:
        if self.value == val:
            return True
        if val < self.value and self.left is not None:
            return self.left.contains(val)
        if val > self.value and self.right is not None:
            return self.right.contains(val)
        return False
