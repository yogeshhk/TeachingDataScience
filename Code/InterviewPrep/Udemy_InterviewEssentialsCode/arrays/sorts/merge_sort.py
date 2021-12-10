# Merge sort
def merge_sort(arr):
    n = len(arr)
    # If the array has less than 2 elements, it's already sorted
    if (n < 2):
        return

    # Sort the left and right halves
    mid = n // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    merge_sort(left_half)
    merge_sort(right_half)

    # Now merge the two halves
    i, j, k = 0, 0, 0
    # Copy data to temp arrays L[] and R[]
    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    # Add any remaining elements
    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1
    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1

arr = [9, 5, -2, 6]
merge_sort(arr)
print(arr)
