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
    ("""2 4""", """1"""),
    ("""3 5""", """7"""),
    ("""20 50""", """573689752"""),
]

if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
