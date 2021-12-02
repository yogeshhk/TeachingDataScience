from typing import List


class Polygon:
    def __init__(self, sides: int):
        self.n = sides
        self.sides = [0] * self.n

    def set_sides(self, sides: List[int]):
        if len(sides) == self.n:
            self.sides = sides

    def get_sides(self) -> List[int]:
        return self.sides
