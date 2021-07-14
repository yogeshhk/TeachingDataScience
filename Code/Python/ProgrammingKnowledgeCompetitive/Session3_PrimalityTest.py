# https://www.youtube.com/watch?v=4UZufg54dFc&list=PLS1QulWo1RIZZc0V_a8cEuFFkF5KbGlsf&index=3
# Primality Test
import math
# Approach 1: O(n)
def approach1(n):
    divcnt = 0
    for i in range(1,n+1):
        if n%i==0:
            divcnt+=1
    return divcnt==2

# Approach 2: Base case + O(1) upto O(root(n))
def approach2(n):
    # Base cases, all O(1)
    if n==0 or n==1:
        return False
    if n==2 or n==3:
        return True
    if n%2 == 0 or n%3 == 0:
        return False
    # non Base case
    for i in range(5,int(math.sqrt(n))+1):
        if n%i == 0 or n%(i+2) == 0:
            return False
    return True


for n in [2,21,5,23]:
    print(f"Primality test of {n} by Approach 1: {approach1(n)}")
    print(f"Primality test of {n} by Approach 2: {approach2(n)}")