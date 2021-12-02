#!/usr/bin/env python3
n = int(input())
canisters = (int(h) for h in input().split())
sorted_canisters = sorted(canisters)

fractions = []
for balloon_size, canister_size in zip(range(1, n + 1), sorted_canisters):
    fractions.append(canister_size / balloon_size)

# Note that the for-loop above can be read like:
# for x in range(n):
#     balloon_size = x + 1
#     canister_size = sorted_canisters[x]
#     fractions.append(canister_size / balloon_size)

largest = max(fractions)
if largest > 1:
    print("impossible")
else:
    smallest = min(fractions)
    print(round(smallest, 16))

# If you're interested, all of the above can be done in two lines of Python code:
# res = [h / b for b, h in zip(range(1, int(input()) + 1), sorted(map(int, input().split())))]
# print("impossible" if max(res) > 1 else f"{min(res):.16f}")
