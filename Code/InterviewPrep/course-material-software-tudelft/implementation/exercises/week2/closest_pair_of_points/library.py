from __future__ import annotations
from math import sqrt
from typing import List


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def distance(self, other: Point) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


def brute_force(xs: List[Point]) -> float:
    best = float("inf")
    if len(xs) < 2:
        return best
    for i in range(len(xs)):
        point = xs[i]
        for j in range(i + 1, len(xs)):
            dist = point.distance(xs[j])
            if dist < best:
                best = dist
    return best


def sort_by_x(xs: List[Point]) -> List[Point]:
    return sorted(xs, key=lambda point: point.x)


def sort_by_y(xs: List[Point]) -> List[Point]:
    return sorted(xs, key=lambda point: point.y)
