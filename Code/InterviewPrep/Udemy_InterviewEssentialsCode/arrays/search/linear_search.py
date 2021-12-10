# Linear search
def linear_search(arr, target):
    for elem in arr:
        if elem == target:
            return True
    return False

arr = [-1, 0, 4, 6, 9]
print(linear_search(arr, 4))
print(linear_search(arr, 0))
print(linear_search(arr, 2))
print(linear_search(arr, 9))
print(linear_search(arr, 10))