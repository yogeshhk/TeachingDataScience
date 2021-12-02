import unittest

from decorators import spectest, yourtest, timeout, solution_only
from weblabTestRunner import TestRunner
from .library import Package, PriorityPackage
from .solution import Warehouse

p1 = Package("Package for T1", 1)
p2 = Package("Package for T2", 2)
p3 = Package("Package for T3", 3)

prio1 = PriorityPackage("Prio package for T1", 1)
prio2 = PriorityPackage("Prio package for T2", 2)
prio3 = PriorityPackage("Prio package for T3", 3)

if solution_only:
    packages = [Package(str(x), x % 3 + 1) for x in range(100000)]
    prio_packages = [PriorityPackage(f"PrioPackage {x}", x % 3 + 1) for x in range(30000)]
    wh_prepared = Warehouse()
    for p in [Package(f"Package {x} is here", x % 3 + 1) for x in range(30000)]:
        wh_prepared.store(p)


class TestSuite(unittest.TestCase):

    @yourtest
    @spectest(1)
    def test_example(self):
        wh = Warehouse()
        wh.store(p1)
        wh.store(p2)
        wh.store(p3)

        self.assertEqual(wh.collect(1), p1)
        self.assertEqual(wh.collect(2), p2)
        self.assertEqual(wh.collect(3), p3)

    @yourtest
    @spectest(1)
    def test_priority(self):
        wh = Warehouse()
        wh.store(p1)
        wh.store(p2)
        wh.store(p3)
        wh.store(prio1)
        wh.store(prio2)
        wh.store(prio3)

        self.assertEqual(wh.collect(1), prio1)
        self.assertEqual(wh.collect(1), p1)
        self.assertEqual(wh.collect(2), prio2)
        self.assertEqual(wh.collect(2), p2)
        self.assertEqual(wh.collect(3), prio3)
        self.assertEqual(wh.collect(3), p3)

    @spectest(1)
    @timeout(0.5)
    def test_efficiency(self):
        wh = Warehouse()
        for p in packages:
            wh.store(p)

        for x in range(100000):
            self.assertEqual(wh.collect(x % 3 + 1), packages[x])

    @spectest(1)
    @timeout(1)
    def test_priority_efficiency(self):
        for p in prio_packages:
            wh_prepared.store(p)
        for x in range(30000):
            self.assertIsInstance(wh_prepared.collect(x % 3 + 1), PriorityPackage)
        for x in range(0, 9):
            self.assertEqual(wh_prepared.collect(x % 3 + 1).content, f"Package {x} is here")


if __name__ == "__main__":
    unittest.main(testRunner=TestRunner)
