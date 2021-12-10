memo = {}
def fib(n):
    if (n not in memo):
        if n <= 1:
            memo[n] = n
        else:
            memo[n] = fib(n-1) + fib(n-2)
    return memo[n]
    
def bad_fib(n):
    if n <= 1:
        return n
    return bad_fib(n-1) + bad_fib(n-2)

def good_fib(n):
    if n <= 1:
        return n
    memo = [0]*(n + 1)
    memo[1] = 1
    for i in range(2, n + 1):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]


for i in range(10):
    print(good_fib(i))

    