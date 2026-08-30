class Point:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self) -> int:
        return self.x + self.y


print(hash(Point(3, 4)))  # Prints 7
print(hash(Point(1, 1)))  # Prints 2
