import subprocess
import unittest

from weblabTestRunner import TestRunner


class TestSolution(unittest.TestCase):

    def test_all(self):
        for test_case, expected in tests:
            p = subprocess.run(['python', 'solution.py'],
                               input=test_case.strip(), stdout=subprocess.PIPE, timeout=1, encoding="ascii")
            self.assertEqual(0, p.returncode, "Solution should not raise an exception")
            self.assertEqual(expected, str(p.stdout).strip(), "Answer should be correct")


tests = [
    ("""5 20
20 50
10 30
5 15
4 12
9 20""", """57"""),
    ("""5 100
20 50
10 30
5 15
4 12
9 20""", """127"""),
]

if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
