# Given number of n integers, are there 3 elements, a,b,c where a+b+c closest to given target.
# Example : [-1,0,1,2,-1,-4] and target 1 => [-1 + 2 + 1] = 2, close to target 1

def threeSumClosest(nums, target):
    best_sum = 1000000
    nums = sorted(nums)

    for i in range(0, len(nums) - 2):
        start = i + 1
        end = len(nums) - 1
        while start < end:
            s = nums[i] + nums[start] + nums[end]
            if s == target:
                return s
            if abs(target - s) < abs(target - best_sum):
                best_sum = s
            if s <= target:
                start += 1
                while nums[start] == nums[start - 1] and start < end:
                    start += 1
            else:
                end -= 1

    return best_sum
