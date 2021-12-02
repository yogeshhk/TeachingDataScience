import unittest

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import closest_pair
from .library import Point
import random


class TestSolution(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_two_points(self):
        p1 = Point(1, 2)
        p2 = Point(4, 6)
        self.assertEqual(5, closest_pair([p1, p2]))

    @yourtest
    @spectest(1)
    def test_small(self):
        points = [Point(x, y) for x, y in
                  [(2.0, 3.0), (12.0, 30.0), (40.0, 50.0), (5.0, 1.0), (12.0, 10.0), (3.0, 4.0)]]
        self.assertEqual(1.4142135623730951, closest_pair(points))

    @spectest(5)
    @timeout(0.1)
    def test_large(self):
        random.seed(201824)
        points = [Point(random.uniform(-50, 90), random.uniform(40, 200)) for _ in range(100)]
        self.assertEqual(1.1880041397906163, closest_pair(points))

    @spectest(5)
    @timeout(3)
    def test_massive(self):
        random.seed(50174)
        points = [Point(random.uniform(-4000, 6000), random.uniform(-3000, 11000)) for _ in range(10000)]
        self.assertEqual(0.25415639590268696, closest_pair(points))

    @spectest(7)
    @timeout(6)
    def test_insane(self):
        random.seed(50174)
        points = [Point(random.uniform(-8000, 9000), random.uniform(-5000, 12000)) for _ in range(30000)]
        self.assertEqual(0.387369335475009, closest_pair(points))

    @spectest(3)
    def test_delta(self):
        points = [Point(x, y) for x, y in
                  [(2.0, 0.0), (-2.0, 2.0), (0.0, 0.0), (0.0, 1.75), (0.0, 2.25), (1.0, 1.0), (3.0, 3.0), (5.0, 3.0)]]
        self.assertEqual(0.5, closest_pair(points))

    @spectest(3)
    def test_border(self):
        points = [Point(x, y) for x, y in
                  [(0.0, 0.0), (-10.0, 0.0), (10.0, 0.0), (-8.0, 6.0), (-8.0, -6.0), (-1.0, 10.0), (1.0, 10.0),
                   (5.0, 3.0), (7.0, -5.0)]]
        self.assertEqual(2, closest_pair(points))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
