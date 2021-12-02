import unittest
from enum import IntEnum
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
                print("Failed at:")
                print(test_case)
                return ResultEnum.RUN_ERROR
            actual = str(p.stdout).strip()
            if actual != expected:
                print("Expected")
                print(expected)
                print("But was")
                print(actual)
                print("For test")
                print(test_case)
                return ResultEnum.WRONG_ANSWER
        except TimeoutExpired:
            print("Failed at:")
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


tests = [
    ("""2 4""", """1"""),
    ("""3 5""", """7"""),
    ("""20 50""", """573689752"""),
    ("""100 200""", """753047707"""),
    ("""100 245""", """717474152"""),
    ("""10 10""", """1"""),
    ("""10 11""", """17"""),
    ("""10 15""", """46305"""),
    ("""10 25""", """415747311"""),
    ("""10 9""", """0"""),
    ("""1 19""", """0"""),
    ("""1 1""", """1"""),
    ("""1 200""", """0"""),
    ("""120 200""", """190784077"""),
    ("""1 2""", """0"""),
    ("""140 200""", """930537670"""),
    ("""160 200""", """693052724"""),
    ("""180 200""", """671098788"""),
    ("""19 1""", """0"""),
    ("""19 2""", """0"""),
    ("""19 4""", """0"""),
    ("""200 200""", """1"""),
    ("""20 200""", """51701172"""),
    ("""2 19""", """1"""),
    ("""2 1""", """0"""),
    ("""2 200""", """1"""),
    ("""284 300""", """405221272"""),
    ("""289 284""", """0"""),
    ("""300 284""", """0"""),
    ("""300 300""", """1"""),
    ("""37 98""", """65568402"""),
    ("""40 200""", """483022842"""),
    ("""60 200""", """902750367"""),
    ("""80 200""", """633231047"""),
]
timeout = 3

if __name__ == "__main__":
    res = run_all_tests()
    unittest.main(testRunner=TestRunner)
