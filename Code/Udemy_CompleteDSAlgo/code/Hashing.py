#   Created by Elshad Karimov 
#   Copyright © 2021 AppMillers. All rights reserved.

def mod(number, cellNumber):
    return number % cellNumber


# print(mod(400, 24))


def modASCII(string, cellNumber):
    total = 0
    for i in string:
        total += ord(i)
    return total % cellNumber

print(modASCII("ABC", 24))
