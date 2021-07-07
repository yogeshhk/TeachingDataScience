# https://www.youtube.com/watch?v=9mBMEIFBgq0&list=PLS1QulWo1RIZZc0V_a8cEuFFkF5KbGlsf&index=2
# 1. All divisors of a number, for 24 divisors are 1,2,3,4,6,8,12,24
# 2. Prime numbers
import math
n = 24
# O(n)
def my_divisors1(n):
    divisors = []
    for i in range(1,n+1):
        if n%i == 0:
            divisors.append(i)
    return divisors

# O(root(n))
def my_divisors2(n):
    divisors = set()
    for i in range(1,int(math.sqrt(n))+1):
        if n%i == 0:
            divisors.add(i)
            divisors.add(n//i)
    return list(divisors)

print(f"Divisors1 of {n} are {my_divisors1(n)}")
print(f"Divisors2 of {n} are {my_divisors2(n)}")