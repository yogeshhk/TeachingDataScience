# Find single number in list having duplicates. O(n)
# Example : [2,2,1] => 1

# with extra memory
def singleNumber_w_memory(nums):
    counts = {}
    for n in nums:
        if n not in counts:
            counts[n] = 1
        else:
            del counts[n]
    return list(counts.keys())[0]

