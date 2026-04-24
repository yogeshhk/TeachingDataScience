# Given an array of integers, return indices of the two numbers such that they add up to a specific target
# Example:
#   nums =  [2,7,11,15], target = 9, as nums[0] + nums[1] = 2 + 7 = 9, return [0,1]
# Ref: https://www.youtube.com/watch?v=sUicrnHwA0s&list=PLiC1doDIe9rDFw1v-pPMBYvD6k1ZotNRO&index=1

# Brute force ie two for loops, so O(n^2)
def two_sum_brute(nums, target):
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return [-1, -1]


# Loop once, dictionary based, so O(n)
def two_sum_dict(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        elif num not in seen:
            seen[num] = i
    return [-1, -1]
