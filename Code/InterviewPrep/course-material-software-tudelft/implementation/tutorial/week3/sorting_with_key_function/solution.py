from decorators import empty
from typing import List
from .library import Point
from operator import attrgetter


@empty
# Returns a sorted list of the points, based on their x coordinate only.
# If two points have the same x coordinate, their respective order is irrelevant.
def order_by_x(xs: List[Point]) -> List[Point]:
    return sorted(xs, key=lambda point: point.x)


@empty
# Returns a sorted list of the points, based on their x coordinate.
# If two points have the same coordinate, their respective order is based on the y coordinate.
def order_by_x_then_y(xs: List[Point]) -> List[Point]:
    # return order_by_x(sorted(xs, key=lambda point: point.y))  # Sort by y first, then x
    return sorted(xs, key=attrgetter("x", "y"))  # Sort based on both attributes in 1 go
