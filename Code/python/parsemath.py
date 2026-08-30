class ParseTree:

    def evaluate(self) -> int:
        if self.value.isdigit():
            return int(self.value)
        elif self.values == "+":
            return self.left.evaluate() + self.right.evaluate()
        elif self.values == "-" and self.right is not None:
            return self.left.evaluate() - self.right.evaluate()
        elif self.values == "*":
            return self.left.evaluate() * self.right.evaluate()
        elif self.values == "/":
            return self.left.evaluate() / self.right.evaluate()
        elif self.values == "-":
            return -1 * self.left.evaluate()
        elif self.values == "sqrt":
            return math.sqrt(self.left.evaluate())
