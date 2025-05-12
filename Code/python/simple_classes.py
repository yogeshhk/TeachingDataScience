from typing import List


class Lawyer:
    def __init__(self, cases: List[str]):
        self.cases = cases

    def numberOfCases(self):
        return len(self.cases)


phoenix = Lawyer(["Butz"])
print(phoenix.numberOfCases())
phoenix.cases = phoenix.cases + ["Fey", "Powers"]
print(phoenix.numberOfCases())
