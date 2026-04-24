def bar(n: int) -> int:
    s = 0
    for i in range(n):
        for j in range(i):
            s += i
    return s
