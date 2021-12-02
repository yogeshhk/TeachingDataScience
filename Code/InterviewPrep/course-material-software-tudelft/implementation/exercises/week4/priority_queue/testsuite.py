import random
import unittest

from decorators import spectest, yourtest, timeout, solution_only
from weblabTestRunner import TestRunner
from .library import Task
from .solution import solve

if solution_only:
    random.seed(691721)
    huge_list = [Task(random.randint(1, 1000)) for _ in range(50000)]


class TestSuite(unittest.TestCase):
    @yourtest
    @spectest(1)
    def test_tasks(self):
        slides = Task(3)
        exams = Task(7)
        assignments = Task(5)
        list_of_tasks = [slides, exams, assignments]
        self.assertEqual(solve(list_of_tasks, 4), 7)

    @spectest(1)
    def test_no_tasks(self):
        self.assertEqual(solve([], 100), 0)

    @spectest(1)
    def test_no_computers(self):
        self.assertEqual(solve([Task(1)], 0), 0)

    @spectest(1)
    def test_equal_number(self):
        self.assertEqual(solve([Task(5), Task(5), Task(5)], 3), 5)

    @spectest(1)
    def test_trivial(self):
        small_list = [Task(i + 1) for i in range(10)]
        self.assertEqual(solve(small_list, 7), 13)

    @spectest(1)
    def test_small(self):
        small_list = [Task(16), Task(8), Task(7), Task(5), Task(16)]
        self.assertEqual(solve(small_list, 2), 32)

    @spectest(1)
    def test_big(self):
        small_list = [Task(i + 1) for i in range(50)]
        self.assertEqual(solve(small_list, 13), 122)

    @spectest(5)
    @timeout(0.2)
    def test_huge(self):
        self.assertEqual(solve(huge_list, 23), 1084119)


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
