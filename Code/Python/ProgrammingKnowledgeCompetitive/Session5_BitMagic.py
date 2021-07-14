# https://www.youtube.com/watch?v=4UZufg54dFc&list=PLS1QulWo1RIZZc0V_a8cEuFFkF5KbGlsf&index=5
# Bit Magic: Bitwise operators: &, |, ~, ^, >>, <<

# and
def evenodd(n):
    result = "even"
    if n&1==1:
        result = "odd"
    return result

for i in [2,4,400,5,67,21]:
    print(f"{i} is {evenodd(i)}")

# right shift, divide power of 2
# n = 200, so n >> 3 is n/2**3 = 25
print(f"200>>3 = {200>>3}")

# left shift, multiply power of 2
# n = 200, so n << 3 is n*2**3 = 1600
print(f"200<<3 = {200<<3}")