# Given number of n integers, are there 3 elements, a,b,c where a+b+c = 0.
# Exmaple : [-1,0,1,2,-1,-4] => [[-1,-1,2],[-1,0,-1]]

def threeSum(nums):
    if len(nums) < 3:
        return []
    nums = sorted(nums)
    triplets = []
    for i in range(0, len(nums) - 2):
        start = i + 1
        end = len(nums) - 1
        while start < end:
            s = nums[i] + nums[start] + nums[end]
            if s == 0:
                triplets.append((nums[i], nums[start], nums[end]))
                start += 1
            elif s < 0:
                start += 1
            else:
                end -= 1
    return list(set(triplets))
