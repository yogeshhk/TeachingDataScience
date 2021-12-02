import random
import unittest

from decorators import spectest, yourtest, timeout
from weblabTestRunner import TestRunner
from .solution import DigitalClock, NegativeTimeError


class TestSolution(unittest.TestCase):

    @spectest(1)
    @yourtest
    @timeout(0.1)
    def test_add_regular(self):
        dc = DigitalClock()
        self.assertTrue(dc.is_ready())  # Time should be 0 before adding the time
        dc.add_time(10)  # Put 10 seconds on the clock

        for x in range(10):
            self.assertFalse(dc.is_ready())  # Time should not be 0
            dc.tick()  # Tick tock

        self.assertTrue(dc.is_ready())  # After 10 ticks time should be 0

    @spectest(1)
    @yourtest
    @timeout(0.1)
    def test_add_negative(self):
        dc = DigitalClock()
        self.assertTrue(dc.is_ready())  # Time should be 0 before adding the time
        self.assertRaises(NegativeTimeError, dc.add_time, -1)  # Adding -1 time should raise NegativeTimeError
        self.assertTrue(dc.is_ready())  # Time should remain unchanged

    @spectest(3)
    @timeout(0.1)
    def test_add_negative_random(self):
        dc = DigitalClock()
        self.assertTrue(dc.is_ready())
        self.assertEqual(0, dc.time)
        random.seed(81531)
        for _ in range(10):
            self.assertRaises(NegativeTimeError, dc.add_time, random.randint(-50, -1))
            self.assertTrue(dc.is_ready())
            self.assertEqual(0, dc.time)

    @spectest(2)
    @timeout(0.1)
    def test_full_behaviour(self):
        dc = DigitalClock()
        random.seed(91654)
        for _ in range(50):
            self.assertEqual(0, dc.time)
            self.assertTrue(dc.is_ready())
            time = random.randint(1, 200)
            dc.add_time(time)
            self.assertEqual(time, dc.time)
            dc.reset()
            self.assertEqual(0, dc.time)
            self.assertRaises(NegativeTimeError, dc.add_time, -time)
            self.assertEqual(0, dc.time)
            self.assertTrue(dc.is_ready())


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
