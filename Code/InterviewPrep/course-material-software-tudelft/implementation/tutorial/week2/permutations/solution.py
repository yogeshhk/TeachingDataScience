from typing import List

from decorators import empty


class PuzzleSolver:
    def __init__(self, a: str, b: str, c: str):
        self.a = a
        self.b = b
        self.c = c
        self.letters = sorted(set(list(self.a) + list(self.b) + list(self.c)))

    # Creates a number from the given string based on the given list
    def string_to_num(self, item: str, xs: List[int]) -> int:
        return int(''.join([str(xs[self.letters.index(char)]) for char in item]))

    # Tests if a given permutation is a correct solution
    def test(self, xs: List[int]) -> bool:
        if len(xs) != len(self.letters):
            return False
        a_num = self.string_to_num(self.a, xs)
        b_num = self.string_to_num(self.b, xs)
        c_num = self.string_to_num(self.c, xs)
        return a_num + b_num == c_num

    @empty
    # Returns a valid solution to the puzzle
    def solve(self) -> List[int]:
        return self.permute(len(self.letters), [], set(range(10)))

    @empty
    # Helper method to permute the numbers
    def permute(self, k: int, s: List[int], u: set) -> List[int]:
        for e in u:
            if k == 1:
                if self.test(s + [e]):
                    return s + [e]
            else:
                p = self.permute(k - 1, s + [e], u - {e})
                if p is not None:
                    return p
