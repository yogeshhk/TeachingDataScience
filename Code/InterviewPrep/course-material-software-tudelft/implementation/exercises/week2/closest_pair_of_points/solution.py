from decorators import empty, remove
from typing import List
from .library import Point, sort_by_x, sort_by_y, brute_force


@empty
# Returns the distance between the closest pair of points
def closest_pair(xs: List[Point]) -> float:
    if len(xs) < 2:
        return float("inf")
    return closest_pair_helper(sort_by_x(xs), sort_by_y(xs))


@remove
def closest_pair_helper(sorted_x: List[Point], sorted_y: List[Point]) -> float:
    if len(sorted_x) < 3:
        return brute_force(sorted_x)
    mid = len(sorted_x) // 2
    left_list = sorted_x[:mid]
    left_sorted = sort_by_y(left_list)
    left = closest_pair_helper(left_list, left_sorted)  # Find the closest pair of points in the left half
    right_list = sorted_x[mid:]
    right_sorted = sort_by_y(right_list)
    right = closest_pair_helper(right_list, right_sorted)  # Find the closest pair of points in the right half
    delta = min(left, right)  # Minimum distance in left and right half separately

    # Now process the points in both halves
    mid_list = []
    mid_x = left_list[-1].x

    # Find all points that are at most delta away from the border between the left and right half
    for point in sorted_y:
        if abs(point.x - mid_x) < delta:
            mid_list.append(point)

    # Compare the points within the center strip
    for i in range(len(mid_list) - 1):
        point1 = mid_list[i]
        for j in range(i + 1, min(i + 12, len(mid_list))):  # Only check the next 11 points in the list
            point2 = mid_list[j]
            dist = point1.distance(point2)
            if dist < delta:
                delta = dist

    return delta
