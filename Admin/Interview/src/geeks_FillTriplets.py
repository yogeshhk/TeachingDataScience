# https://www.geeksforgeeks.org/find-triplets-array-whose-sum-equal-zero/

# Given an array of distinct elements. The task is to find triplets in the array whose sum is zero
# Brute force: 3 for loops, O(n^3)

def is_duplicate(i, j, k, triplets):
    if i == j or j == k or k == i:
        return True
    for tpl in triplets:
        if i in tpl and j in tpl and k in tpl:
            return True
    return False


def find_triplets_bruteforce(nums):
    triplets = []
    for i in nums:
        for j in nums:
            for k in nums:
                if i + j + k == 0 and not is_duplicate(i, j, k, triplets):
                    triplets.append((i, j, k))
    return triplets


# Slightly better approach, two for loops and stored set of numbers.
# Take sum of two numbers, then check if remaining number is in the set or not

def find_triplets_better(nums):
    triplets = []
    nums_set = set(nums)
    for i in nums:
        for j in nums:
            remaining = -1 * (i + j)
            if remaining in nums_set and not is_duplicate(i, j, remaining, triplets):
                triplets.append((i, j, remaining))
    return triplets


arr = [0, -1, 2, -3, 1]
# arr = [1, -2, 1, 0, 5]
# output = find_triplets_bruteforce(arr)
output = find_triplets_better(arr)
print(output)
