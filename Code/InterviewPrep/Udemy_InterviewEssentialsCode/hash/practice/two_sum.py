def two_sum(nums, target):
    seen = set()
    for num in nums:
        compliment = target - num
        if compliment in seen:
            return True
        else:
            seen.add(num)
    return False

print(two_sum([5, 2, -1, 10, 8], 10))
print(two_sum([5, 2, -1, 10, 8], 1))
print(two_sum([5, 2, -1, 10, 8], 3))