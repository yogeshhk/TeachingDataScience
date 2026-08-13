# Place 4 chess queens in 4x4 grid so that no one gets killed
# Only one queen can be in one row, so need to store non-eligible
# columns only apart from Positive Diagonal and Negative Diagonal positios
# "." means empty

def solve_queens(n):
    columns_set = set()
    positive_diagonal_set = set() # row + col = same
    negative_diagonal_set = set() # row - col = same

    result = []
    board = [["."] for i in range(n)]

    # nested function has access to callee-function's variables
    def place_queen(row):
        if row == n - 1:
            copy_board = ["".join(row) for row in board]
            result.append(copy_board)
            return
        for col in range(n):
            if col in columns_set or row +col in positive_diagonal_set or row - col in negative_diagonal_set:
                continue
            columns_set.add(col)
            positive_diagonal_set.add(row + col)
            negative_diagonal_set.add(row - col)
            board[row][col] = "Q"

            place_queen(row + 1) #recursion, next positon

            # need to reset additions done before, so that data is restored
            # That's BACKTRACKING
            columns_set.remove(col)
            positive_diagonal_set.remove(row + col)
            negative_diagonal_set.remove(row - col)
            board[row][col] = "."

    # call
    place_queen(0)
    return result


result = solve_queens(4)
print(result)





