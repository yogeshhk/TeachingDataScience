def mia(n: int) -> int:
    s = 0
    for i in range(n):
        for j in range(n):
            s += (i * j) ** 1.5
    return s
