#   Created by Elshad Karimov 
#   Copyright © AppMillers. All rights reserved.

# Number of paths to reach the last cell with given cost in 2D array

def numberOfPaths(twoDArray, row, col, cost):
    if cost < 0:
        return 0
    elif row == 0 and col == 0:
        if twoDArray[0][0] - cost == 0:
            return 1
        else:
            return 0
    elif row == 0:
        return numberOfPaths(twoDArray, 0, col-1, cost - twoDArray[row][col] )
    elif col == 0:
        return numberOfPaths(twoDArray, row-1, 0, cost - twoDArray[row][col] )
    else:
        op1 = numberOfPaths(twoDArray, row -1, col, cost - twoDArray[row][col] )
        op2 = numberOfPaths(twoDArray, row, col-1, cost - twoDArray[row][col] )
        return op1 + op2


TwoDList = [[4,7,1,6],
           [5,7,3,9],
           [3,2,1,2],
           [7,1,6,3]
           ]

print(numberOfPaths(TwoDList, 3,3, 25))