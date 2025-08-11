# Find majority element from an array, and has more than half the length
# Example [3,2,3] => 3

def majorityElement(nums):
    sums = {}
    for n in nums:
        if n not in sums:
            sums[n] = 1
        else:
            sums[n] += 1
        if sums[n] > len(nums) / 2:
            return n
