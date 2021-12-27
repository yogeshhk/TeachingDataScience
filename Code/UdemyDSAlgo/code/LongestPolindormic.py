#   Created by Elshad Karimov 
#   Copyright © AppMillers. All rights reserved.

# Longest Polindromic Subsequence in Python

def findLPS(s, startIndex, endIndex):
    if startIndex > endIndex:
        return 0
    elif startIndex == endIndex:
        return 1
    elif s[startIndex] == s[endIndex]:
        return 2 + findLPS(s, startIndex+1, endIndex-1)
    else:
        op1 = findLPS(s, startIndex, endIndex-1)
        op2 = findLPS(s, startIndex+1, endIndex)
        return max(op1, op2)

print(findLPS("ELRMENMET", 0, 8))