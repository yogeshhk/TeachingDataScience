# Ways in which you can cling n steps. You can take max 2 steps at a time.
# Example For 2, ways = 2 (1 +1, 2)
def climbStairs(n):
    path = {1:1,2:2,3:3}
    for x in range(4,n+1):
        path[x] = path[x-1] + path[x-2]
    return path[n]
