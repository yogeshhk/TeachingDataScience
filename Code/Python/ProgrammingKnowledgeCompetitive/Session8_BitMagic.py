# https://www.youtube.com/watch?v=4UZufg54dFc&list=PLS1QulWo1RIZZc0V_a8cEuFFkF5KbGlsf&index8
# Bit Magic: Bitwise operators: &, |, ~, ^, >>, <<

# to binary
def int2bin(n):
    return str(bin(n))[2:] # about starting '0b'

def bin2int(s):
    return int(s,2)

# if kth bit set
def kthbitset(n,k):
    if n & (1 << (k-1)):
        return "set"
    return "not set"

for i in [2,4,400,5,67,21,1024]:
    tobin = int2bin(i)
    backtoint = bin2int(tobin)
    print(f"{i} : bin {tobin} int {backtoint}, 3rd bit set? {kthbitset(i,3)}")

# Find numbers which occur once. n^0=n, n^n =0
nums = [5,3,2,3,1,2,3,5,7]
def finduniques(arry):
    res = arry[0]
    for i in range(1, len(arry)):
        res = res ^ arry[i]
    return  res

print(f"Unique nums in {nums} are {finduniques(nums)}") # Does not give '7'!!
