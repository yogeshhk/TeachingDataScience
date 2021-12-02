import unittest

from decorators import spectest, yourtest
from weblabTestRunner import TestRunner
from .library import Polygon
from .solution import Triangle


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_area(self):
        t = Triangle()
        t.set_sides([3, 4, 5])
        self.assertEqual(t.find_area(), 6)

    @spectest(5)
    def test_instance(self):
        t = Triangle()
        self.assertTrue(isinstance(t, Polygon))

    @spectest(5)
    def test_sides_num(self):
        t = Triangle()
        self.assertEqual(len(t.get_sides()), 3)

    @spectest(1)
    def test_area1(self):
        t = Triangle()
        t.set_sides([2, 3, 4])
        self.assertEqual(t.find_area(), 2.9)

    @spectest(1)
    def test_area2(self):
        t = Triangle()
        t.set_sides([23, 10, 32])
        self.assertEqual(t.find_area(), 58.9)

    @spectest(1)
    def test_area3(self):
        t = Triangle()
        t.set_sides([10, 20, 20])
        self.assertEqual(t.find_area(), 96.8)

    @spectest(1)
    def test_area4(self):
        t = Triangle()
        t.set_sides([13, 13, 13])
        self.assertEqual(t.find_area(), 73.2)

    @spectest(1)
    def test_area5(self):
        t = Triangle()
        t.set_sides([7, 13, 8])
        self.assertEqual(t.find_area(), 24.2)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
