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
    ("""3 10
WordMispronounced LoudMic Gaming
Sleeping FreeLunch Latecomer
2MinutesSilence TeacherAngry Eating
MicBreaks
Sleeping
SlideshowContainsMistake
WordMispronounced
Gaming
PhoneRings
Eating
Latecomer
TeacherAngry
2MinutesSilence""", """7"""),
    ("""3 5
EventA EventB EventC
EventD EventE EventF
EventG EventH EventI
EventJ
EventA
EventK
EventB
EventD""", """:-("""),
]

if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
