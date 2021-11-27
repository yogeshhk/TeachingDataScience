def maya(n: int) -> int:
    x = list()
    for i in range(n):
        for j in range(n):
            x += [i * j]
    s = 0
    for i in x:
        s += i ** 1.5
    return s
