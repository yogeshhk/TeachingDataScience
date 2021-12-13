#   Created by Elshad Karimov 
#   Copyright © AppMillers. All rights reserved.

# recursiveRange Solution

def recursiveRange(num):
    if num <= 0:
        return 0
    return num + recursiveRange(num - 1)


print(recursiveRange(6))