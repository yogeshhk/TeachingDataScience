
def bubble_sort(nums):
    nums = list(nums) # copy
    for i in range(len(nums) -1 ):
        for j in range(len(nums) -1 ):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums

def merge(nums1, nums2):
    merged = []
    i,j=0,0
    while i<len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            merged.append(nums2[j])
            j+=1
    nums1_tail = nums1[i:]
    nums2_tail = nums2[j:]
    return merged + nums1_tail + nums2_tail

def merge_sort(nums):
    nums = list(nums) # copy
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left = nums[:mid]
    right = nums[mid:]
    left_sorted, right_sorted = merge_sort(left), merge_sort(right)
    sorted_nums = merge(left_sorted, right_sorted)
    return sorted_nums

array = [1,3,22,-55,66,33]
print(array)
print(bubble_sort(array))
print(merge_sort(array))
