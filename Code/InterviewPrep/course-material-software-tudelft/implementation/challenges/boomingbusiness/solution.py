# Solution to: Bonsai
# Using a more clever recursion, adding one branch at a time.
# Runtime: O(W^2)
# Created by Raymond for BAPC 2017

MOD = 1000000007

input_list = input().split(" ")
h = int(input_list[0])
w = int(input_list[1])

# memory_a[i][j] = number of Bonsai trees of left-height i and weight j, never exceeding height h
memory_a = [x[:] for x in [[0] * (w + 1)] * (h + 1)]
# memory_b[i][j] = number of Bonsai trees of left-height i and weight j, never exceeding height h-1
memory_b = [x[:] for x in [[0] * (w + 1)] * (h + 1)]

# This if-statement added by Mees to avoid errors.
if h == 1:
    print(1 if w == 1 else 0)
    exit()

memory_a[1][1] = memory_b[1][1] = 1

# Start Dynamic Programming
# memory_a[i][j] = sum_{k=i-1}^h     memory_a[k][j-1]
# memory_b[i][j] = sum_{k=i-1}^{h-1} memory_b[k][j-1]
# i.e. in order to make a tree of weight j and left-height i
# we take a tree of left-height k at least i and weight j-1,
# and we attach a branch on the left, making the new tree have
# exactly left-height i.
for j in range(2, w + 1):
    sum_a = memory_a[h][j - 1]
    sum_b = memory_b[h - 1][j - 1]
    for i in range(h, 1, -1):
        sum_a += memory_a[i - 1][j - 1]
        sum_a %= MOD
        memory_a[i][j] += sum_a
        memory_a[i][j] %= MOD
    for i in range(h - 1, 1, -1):
        sum_b += memory_b[i - 1][j - 1]
        sum_b %= MOD
        memory_b[i][j] += sum_b
        memory_b[i][j] %= MOD

ans = 0
for i in range(1, h + 1):
    ans += MOD + memory_a[i][w] - memory_b[i][w]
    ans %= MOD

print(ans)
