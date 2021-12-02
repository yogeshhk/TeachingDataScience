In this exercise you will implement a solution to the closest pair of points problem.

Given a list of points, determine the minimum distance between any 2 points in O(n log n) time.

In the library you will find several things:
- The `Point` class
    - The `distance` function that determines the distance between the current point and another point
- A `brute_force` solution to the problem (compares all points to all other points, an O(n^2) solution)
- Two functions `sort_by_x` and `sort_by_y` that return the given list of points sorted on x or y coordinate respectively
