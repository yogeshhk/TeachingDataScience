from typing import Tuple, Dict

# Read input: n, m, the grid of squares, and the list of events
line = input().split(" ")
n = int(line[0])
m = int(line[1])
grid = [input().split(" ") for _ in range(n)]
events = [input() for _ in range(m)]

# Trivial base case: only a free square if n == 1!
if n == 1:
    print("0")
    exit()

# Create a map from events to x,y-positions
square_map: Dict[str, Tuple[int, int]] = {}
for x in range(n):
    for y in range(n):
        square_map[grid[x][y]] = x, y

# Counters for each row, column, and diagonal, indicating how many squares have been crossed
rows = [0 for _ in range(n)]
rows[n // 2] = 1
cols = [0 for _ in range(n)]
cols[n // 2] = 1
diags = [1, 1]


# Convenience function that prints the result and aborts the code
def done(i):
    print(i + 1)
    exit()


# For all events:
for i, event in enumerate(events):
    # If we don't have it on our grid, skip the event
    if event not in square_map:
        continue
    # Get the position of the square on our grid
    x, y = square_map[event]
    # Increase the counter for row x, if it's full, we're done
    rows[x] += 1
    if rows[x] == n:
        done(i)
    # Increase the counter for column y, if it's full, we're done
    cols[y] += 1
    if cols[y] == n:
        done(i)
    # Check if the square is on the main diagonal
    if x == y:
        diags[0] += 1
        if diags[0] == n:
            done(i)
    # Check if the square is on the other diagonal
    if x == n - y - 1:
        diags[1] += 1
        if diags[1] == n:
            done(i)

# We're out of events, but don't have "BINGO!", so output a sad face
print(":-(")
