from decorators import empty
from .library import Peg


@empty
# This function solves the "Towers of Hanoi" problem.
# It receives as input a number `n` of disks and three
# pegs `source`, `helper`, `target`.
def hanoi(n: int, source: Peg, helper: Peg, target: Peg):
    if n > 0:
        hanoi(n - 1, source, target, helper)
        if source:
            target.push(source.pop())
        hanoi(n - 1, helper, source, target)
