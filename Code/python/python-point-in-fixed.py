class Point:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False


x = [Point(3, 5), Point(1, 4), Point(2, 3)]
p = Point(3, 5)
print(p in x)
