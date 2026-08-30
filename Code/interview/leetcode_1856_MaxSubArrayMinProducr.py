# The min-product of an array is equal to the minimum value in the array multiplied by the array's sum.
#
# For example, the array [3,2,5] (minimum value is 2) has a min-product of 2 * (3+2+5) = 2 * 10 = 20. Given an array
# of integers nums, return the maximum min-product of any non-empty subarray of nums. Since the answer may be large,
# return it modulo 109 + 7.
#
# Note that the min-product should be maximized before performing the modulo operation. Testcases are generated such
# that the maximum min-product without modulo will fit in a 64-bit signed integer.
#
# A subarray is a contiguous part of an array.
#
#
#
# Example 1:
#
# Input: nums = [1,2,3,2]
# Output: 14
# Explanation: The maximum min-product is achieved with the subarray [2,3,2] (minimum value is 2).
# 2 * (2+3+2) = 2 * 7 = 14.

# Hint: Use monotonically increasing stack. If 1 is min, how long can you stretch it
#[1,2]
# for "1", next is 2, so increasing, add it, 1 is still a min, go on doing that it you get a smaller than 1
# for starting with "2", go on doing the same. If we get decreasing, pop th stack

def max_sum_min_product(nums):
    result = 0
    stack = []
    prefix = [0]
    for n in nums:
        prefix.append(prefix[-1] + n)

    for i, n in enumerate(nums):
        newStart = i
        while stack and stack[-1][1] > n:
            start, val = stack.pop()
            total = prefix[i] - prefix[start]
            result = max(result, val*total)
            newStart = start
        stack.append((newStart,n))
    for start, val in stack:
        total = prefix[len(nums)] - prefix[start]
        result = max(result, val*total)

    return result % (10**9 + 7)
