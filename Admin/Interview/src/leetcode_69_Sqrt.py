# Compute Square root
# Example for 4, its 2

def mySqrt(x):
    if x <= 1:
        return x
    else:
        x_n = 0.5 * x  # some estimate
        change = 1
        while change > 0.01:
            next_n = 0.5 * (x_n + x / x_n)
            change = abs(x_n - next_n)
            x_n = next_n
        return int(x_n)
