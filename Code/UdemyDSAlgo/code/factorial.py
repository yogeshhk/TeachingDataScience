#   Created by Elshad Karimov 
#   Copyright © AppMillers. All rights reserved.

# Factorial Solution

def factorial(num):
    if num <= 1:
        return 1
    return num * factorial(num-1)


print(factorial(1)) # 1
print(factorial(2)) # 2
print(factorial(4)) # 24
print(factorial(7)) # 5040