# Binary search
def binary_search(arr, target):
    start = 0
    end = len(arr) - 1

    while (start <= end):
        mid = (start + end) // 2
        if arr[mid] < target:
            # Too small - search right
            start = mid + 1
        elif arr[mid] > target:
            # Too big - search left
            end = mid - 1
        else:
            # Found the value
            return True
    return False

arr = [-1, 0, 4, 6, 9]
print(binary_search(arr, 4))
print(binary_search(arr, 0))
print(binary_search(arr, 2))
print(binary_search(arr, 9))
print(binary_search(arr, 10))
