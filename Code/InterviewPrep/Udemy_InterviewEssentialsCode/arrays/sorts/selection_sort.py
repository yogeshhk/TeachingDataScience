# Selection sort
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        # Find the index of the minimum remaining element
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        # Swap with the first unsorted element
        print("Swapping " + str(arr[min_index]) + " and " +
              str(arr[i]))
        arr[i], arr[min_index] = arr[min_index], arr[i]

arr = [9, 5, -2, 6]
selection_sort(arr)
print(arr)


