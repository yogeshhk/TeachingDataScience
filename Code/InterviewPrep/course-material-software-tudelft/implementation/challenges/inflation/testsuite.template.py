import subprocess
import unittest

from weblabTestRunner import TestRunner


class TestSolution(unittest.TestCase):

    def test_all(self):
        for test_case, expected in tests:
            p = subprocess.run(['python', 'solution.py'],
                               input=test_case.strip(), stdout=subprocess.PIPE, timeout=1, encoding="ascii")
            self.assertEqual(0, p.returncode, "Solution should not raise an exception")
            actual = str(p.stdout).strip()
            if expected == "impossible":
                self.assertEqual(expected, actual, "Answer should be correct")
            else:
                self.assertAlmostEqual(float(expected), float(actual),
                                       msg="Answer should be correct", delta=1e-6)


tests = [
    ("""6
6 1 3 2 2 3""", """0.6"""),
    ("""2
2 2""", """impossible"""),
    ("""5
4 0 2 1 2""", """0"""),
]

if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
