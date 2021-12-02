import random
import unittest
from enum import IntEnum
from math import isclose
from subprocess import run, PIPE, TimeoutExpired

from weblabTestRunner import TestRunner


class ResultEnum(IntEnum):
    RUN_ERROR = 0
    WRONG_ANSWER = 1
    TIMEOUT = 2
    CORRECT = 3


def run_all_tests() -> ResultEnum:
    for test_case, expected in tests:
        try:
            p = run(['python', 'solution.py'], input=test_case.strip(), stdout=PIPE, timeout=1, encoding="ascii")
            if p.returncode != 0:
                print("Run error at:")
                print(test_case)
                return ResultEnum.RUN_ERROR
            actual = str(p.stdout).strip()
            if (expected == "impossible") != (actual == "impossible") or \
                    actual != "impossible" and not isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6):
                print("Wrong answer at:")
                print(test_case)
                print("Expected")
                print(expected)
                print("But was")
                print(actual)
                return ResultEnum.WRONG_ANSWER
        except TimeoutExpired:
            print("Timeout at:")
            print(test_case)
            return ResultEnum.TIMEOUT
    return ResultEnum.CORRECT


# SPECTESTS START HERE
class TestSolution(unittest.TestCase):

    def test_error(self):
        if res < 1:
            self.fail(res)

    def test_wrong(self):
        if res < 2:
            self.fail(res)

    def test_timeout(self):
        if res < 3:
            self.fail(res)


# SPECTESTS END HERE

random.seed(2615897)

random_impossible = list(range(1, 119709))
random.shuffle(random_impossible)
random_impossible[38461] += 1

tests = [
    ("""6
6 1 3 2 2 3""", """0.6"""),
    ("""2
2 2""", """impossible"""),
    ("""5
4 0 2 1 2""", """0"""),
    ("""1
1""", """1"""),
    ("""1
0""", """0"""),
    ("""5
0 0 0 5 5""", """impossible"""),
    ("""60
1 2 3 4 5 4 5 5 9 10 9 10 8 13 10 10 9 18 19 19 21 22 20 24 16 21 17 15 28 25 31 17 17 20 22 25 34 31 37 23 33 35 22 \
42 30 27 29 27 35 49 33 40 39 38 39 56 44 58 32 48""", """0.625"""),
    ("""100
""" + " ".join([str(i) for i in range(100, 0, -1)]), """1"""),
    ("""200000
""" + " ".join([str(i) for i in range(1, 200001)]), """1"""),
    ("""200000
""" + " ".join(["1" for i in range(1, 200001)]), """0.000005"""),
    ("""119708
""" + " ".join([str(i + 1 if i == 84636 else i) for i in random_impossible]), """impossible"""),
    ("""119305
""" + " ".join([str(random.randint(1, i)) for i in range(1, 119306)]), """0.0952380952380952"""),
    ("""200000
""" + " ".join([str(random.randint(int(2*i/3)+1, i)) for i in range(1, 200001)]), """0.7142857142857142857"""),
    ("""200000
""" + " ".join([str(random.randint(1, 200001)) for i in range(1, 200001)]), """impossible"""),
    ("""200000
""" + " ".join(["200000" for i in range(1, 200001)]), """impossible"""),
]
timeout = 5

if __name__ == "__main__":
    res = run_all_tests()
    unittest.main(testRunner=TestRunner)
