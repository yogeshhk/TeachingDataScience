Summation puzzles are a mathematical equation where numbers are replaced by letters. You solve them by
uniquely matching each letter to a number between 0 and 9 such that the equation holds.

Some examples are:

- pot + pan = bib
- dog + cat = pig
- boy + girl = baby

#### Your task
Implement the given methods in the `PuzzleSolver` class so that `solve()` returns a correct solution for similar problems.
The input is three words, which represent an equation in the form `a + b = c`.
The words will never have more than 10 distinct characters and every character must be uniquely mapped to a number 0-9.

#### Implementation details
The three words are stored by `__init__`.
Additionally an alphabetically sorted list of all distinct letters in the words is stored:`self.letters`.
Your solution list will be mapped to these letters in the same order to test for correctness.

For the first example, the function `solve` could output `[0, 2, 4, 5, 3, 1, 7]`,
as a = 0, b = 2, i = 4, n = 5, o = 3, p = 1, t = 7 is a correct solution to the problem.
You can test this manually by replacing all letters by the given numbers.

To determine if a solution is correct, you can call the `test` function with your possible solution.

Credits: this assignment is based on exercise P4.24 from 'Data Structures & Algorithms in Python' by Goodrich et al.
