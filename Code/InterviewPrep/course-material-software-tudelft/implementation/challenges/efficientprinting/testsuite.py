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
    ("""1""", """0"""),
    ("""7""", """1"""),
    ("""42""", """9"""),
    ("""0""", """0"""),
    ("""1000000000000000000""", """249999999999999995"""),
    ("""1000""", """249"""),
    ("""100""", """24"""),
    ("""10""", """2"""),
    ("""123456789""", """30864192"""),
    ("""124""", """28"""),
    ("""125""", """31"""),
    ("""126""", """31"""),
    ("""1953124""", """488272"""),
    ("""1953125""", """488281"""),
    ("""1953126""", """488281"""),
    ("""1""", """0"""),
    ("""24""", """4"""),
    ("""25""", """6"""),
    ("""26""", """6"""),
    ("""298023223876953124""", """74505805969238256"""),
    ("""298023223876953125""", """74505805969238281"""),
    ("""298023223876953126""", """74505805969238281"""),
    ("""2""", """0"""),
    ("""3124""", """776"""),
    ("""3125""", """781"""),
    ("""3216""", """802"""),
    ("""3814697265624""", """953674316388"""),
    ("""3814697265625""", """953674316406"""),
    ("""3814697265626""", """953674316406"""),
    ("""3""", """0"""),
    ("""42""", """9"""),
    ("""4""", """0"""),
    ("""5""", """1"""),
    ("""624""", """152"""),
    ("""625""", """156"""),
    ("""626""", """156"""),
    ("""6""", """1"""),
    ("""7""", """1"""),
    ("""8""", """1"""),
    ("""987654321987654321""", """246913580496913566"""),
    ("""9""", """1"""),
]
timeout = 1

if __name__ == "__main__":
    res = run_all_tests()
    unittest.main(testRunner=TestRunner)
