from collections import defaultdict


class Point:

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __hash__(self) -> int:
        return self.x * 31 + self.y * 7


print(hash(Point(3, 4)))  # Prints 121
print(hash(Point(1, 1)))  # Prints 38


d = defaultdict(int)
for i in range(1000):
    for j in range(1000):
        d[hash(Point(i, j)) % 100] += 1

print(len(d), d)
