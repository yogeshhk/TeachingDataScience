#!/usr/bin/python3

# First, we read in `n` and `m`, which are the same for both solutions.
line = input().split(" ")
n = int(line[0])
m = int(line[1])


# The first solution uses one O(nm) array to store intermediate results.
def quadratic_space_solution(n: int, m: int) -> int:
    # initialize an (n + 1) x (m + 1) array with zeroes
    memory = [x[:] for x in [[0] * (m + 1)] * (n + 1)]

    for i in range(1, n + 1):
        # Read in the price and happiness of the current beer
        line = input().split(" ")
        p = int(line[0])
        h = int(line[1])

        for j in range(1, m + 1):
            memory[i][j] = memory[i - 1][j]
            if j >= p and memory[i][j] < memory[i - 1][j - p] + h:
                memory[i][j] = memory[i - 1][j - p] + h

    return memory[n][m]


# The following solution uses two O(m) arrays that are swapped every time, to save space.
def linear_space_solution(n: int, m: int) -> int:
    # Initialize two arrays of size (m + 1) with zeroes
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        # Swap the two arrays
        prev, curr = curr, prev

        # Read in the price and happiness of the current beer
        line = input().split(" ")
        p = int(line[0])
        h = int(line[1])

        for j in range(1, m + 1):
            curr[j] = prev[j]
            if j >= p and curr[j] < prev[j - p] + h:
                curr[j] = prev[j - p] + h

    return curr[m]


# We only use one of the two solutions to print the final answer.
print(linear_space_solution(n, m))
