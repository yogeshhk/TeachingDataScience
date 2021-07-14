# https://www.youtube.com/watch?v=4UZufg54dFc&list=PLS1QulWo1RIZZc0V_a8cEuFFkF5KbGlsf&index=4
# Prime number generation by Sieve Erotosthenes

import math
def primality(n):
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

# Approach 1: O(n*roont_n), traverse all and check each for primality
def approach1(n):
    primes = []
    for i in range(1,n+1):
        if primality(i):
            primes.append(i)
    return primes

# Approach 2: Sieve theorem
# Upto n, remove all even numbers except 2, then all multiples of 3 but 3
# and then 5...remove all 5 divisible ones and greater than 25

def approach2(n):
    primes_masks = [True]*(n+1)
    primes_masks[0] = False
    primes_masks[1] = False
    for p in range(2,int(math.sqrt(n))+1):
        if primes_masks[p] == True:
            for i in range(p*p,n+1,p):
                primes_masks[i] = False
    primes = [i for i,n in enumerate(primes_masks) if i>1 and n == True]
    return primes


for n in [2,21,5,23]:
    print(f"Primality test of {n} by Approach 1: {approach1(n)}")
    print(f"Primality test of {n} by Approach 2: {approach2(n)}")
