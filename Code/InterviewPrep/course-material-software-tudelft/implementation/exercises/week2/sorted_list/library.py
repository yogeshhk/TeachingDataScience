from typing import List


class Wallet:
    notes: List[int]

    def __init__(self):
        self.notes = []

    def is_empty(self) -> bool:
        return len(self.notes) == 0
