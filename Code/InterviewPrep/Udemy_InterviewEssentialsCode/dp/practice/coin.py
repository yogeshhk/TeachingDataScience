def coin_change(coins, amount):
    memo = [0] + [float('inf') for i in range(amount)]
        
    for i in range(1, amount + 1):
        for coin in coins:
            if i - coin >= 0:
                memo[i] = min(memo[i], memo[i - coin]) + 1

    if memo[-1] == float('inf'):
        return -1
    return memo[-1]

print(coin_change([1, 2, 5], 11)) # 3
print(coin_change([2], 3)) # -1
