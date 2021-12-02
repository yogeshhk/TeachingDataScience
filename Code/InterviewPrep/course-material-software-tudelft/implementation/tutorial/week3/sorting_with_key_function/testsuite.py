import unittest
import random
from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from operator import attrgetter
from .solution import order_by_x, order_by_x_then_y
from .library import Point


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example_x(self):
        random.seed(42)
        points = []
        for x in range(10):
            for y in range(10):
                points.append(Point(x, y))
        random.shuffle(points)
        sol = order_by_x(points)
        for x in range(9, -1, -1):
            for _ in range(10):
                self.assertEqual(sol.pop().x, x)

    @yourtest
    @spectest(1)
    def test_example_x_and_y(self):
        random.seed(42)
        points = []
        for x in range(10):
            for y in range(10):
                points.append(Point(x, y))
        test = points.copy()
        random.shuffle(test)
        self.assertEqual(order_by_x_then_y(test), points)

    @spectest(2)
    @timeout(1)
    def test_large_x(self):
        random.seed(15064)
        points = []
        for x in range(0, 1000, 3):
            for y in range(100):
                points.append(Point(x, y))
        random.shuffle(points)
        sol = order_by_x(points)
        for x in range(999, -1, -3):
            for _ in range(100):
                self.assertEqual(sol.pop().x, x)

    @spectest(3)
    @timeout(2)
    def test_large_x_and_y(self):
        random.seed(59194)
        points = []
        for x in range(-1000, 1000, 5):
            for y in range(-100, 100, 3):
                points.append(Point(x, y))
        test = points.copy()
        random.shuffle(test)
        self.assertEqual(order_by_x_then_y(test), points)

    @spectest(2)
    @timeout(3)
    def test_random_x(self):
        random.seed(39174)
        points = []
        for x in range(300):
            for y in range(300):
                points.append(Point(random.randint(-50, 50), random.randint(-50, 50)))
        random.shuffle(points)
        sol = sorted(points, key=lambda circle: circle.x)
        stud = order_by_x(points)
        for i in range(len(points)):
            self.assertEqual(sol[i].x, stud[i].x)

    @spectest(2)
    @timeout(3)
    def test_random_x_and_y(self):
        random.seed(58441)
        points = []
        for x in range(300):
            for y in range(300):
                points.append(Point(random.randint(-50, 50), random.randint(-50, 50)))
        random.shuffle(points)
        self.assertEqual(order_by_x_then_y(points), sorted(points, key=attrgetter("x", "y")))


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
