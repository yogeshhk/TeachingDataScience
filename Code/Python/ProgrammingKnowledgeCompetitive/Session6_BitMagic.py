# https://www.youtube.com/watch?v=4UZufg54dFc&list=PLS1QulWo1RIZZc0V_a8cEuFFkF5KbGlsf&index=6
# Bit Magic: Bitwise operators: &, |, ~, ^, >>, <<

# and, not
def ispowerof2(n):
    x = n
    y = not(n & (n-1))
    return x and y

for i in [2,4,400,5,67,21,1024]:
    print(f"{i} is {ispowerof2(i)}")

def countonebits1(n):
    s = str(bin(n))[2:]
    return s.count('1')

def countonebits2(n):
    count = 0
    while n:
        count +=1
        n = n & (n-1)
    return count


for i in [2, 4, 400, 5, 67, 21, 1024]:
    print(f"{i} has count of 1s as {countonebits1(i)} also {countonebits2(i)}")