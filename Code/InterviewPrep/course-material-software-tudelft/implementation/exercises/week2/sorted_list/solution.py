import decorators
from .library import Wallet


class RaysWallet(Wallet):

    # The constructor of the class.
    def __init__(self):
        Wallet.__init__(self)

    @decorators.empty
    # A function taking as input a note
    # and assigning it to the correct position
    # in the wallet according to its value.
    def add(self, note: int) -> None:
        n = self.notes
        for i in range(len(self.notes)):
            if note > n[i]:
                n.insert(i, note)
                return
        self.notes.append(note)

    @decorators.empty
    # A function taking as input a note
    # and removing the first occurrence
    # of that note value if there is any.
    def remove(self, note: int) -> None:
        for i in range(len(self.notes)):
            if note == self.notes[i]:
                self.notes.remove(self.notes[i])
                return

    @decorators.empty
    # A function taking as input the wanted balance
    # and returning true if there is enough money in the
    # wallet to cover it or false otherwise.
    def contains(self, balance: int) -> bool:
        if balance == 0:
            return True
        if self.is_empty():
            return False
        amount = 0
        for i in range(len(self.notes)):
            amount += self.notes[i]
            if amount >= balance:
                return True
        return False

    @decorators.empty
    # A function removing the first occurrence of the provided
    # as input note and moving all other occurrences of the same
    # value to the front of the wallet.
    def remove_mtf(self, note: int) -> None:
        for i in range(len(self.notes)):
            if note == self.notes[i]:
                self.notes.remove(self.notes[i])
                while i < len(self.notes) and note == self.notes[i]:
                    del self.notes[i]
                    self.notes.insert(0, note)
                    i += 1
                return
