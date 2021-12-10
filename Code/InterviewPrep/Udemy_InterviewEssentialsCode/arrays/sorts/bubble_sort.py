# Bubble Sort
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if (arr[j] > arr[j + 1]):
                print("Swapping " + str(arr[j]) + " and " +
                      str(arr[j + 1]))
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

arr = [9, 5, -2, 6]
bubble_sort(arr)
print(arr)


