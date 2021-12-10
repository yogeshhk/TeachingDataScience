def search(nums, target):
    start = 0
    end = len(nums)
    leftNum = nums[0]
    rightNum = nums[-1]
    while (start <= end):
        mid = (start + end) // 2

        if nums[mid] == target:
            return True
        else:
            if nums[mid] > rightNum:
                # you are on the left of the line
                if target >= leftNum and target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            else:
                # you are on the right of the line
                if target <= rightNum and target > nums[mid]:
                    start = mid + 1
                else:
                    end = mid - 1
    return False
  
print(search([4, 5, 0, 1, 2, 3], 3))
print(search([4, 5, 0, 1, 2, 3], 6))