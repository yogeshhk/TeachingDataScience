def two_sum_sorted(arr, target):
  # Use the two pointer method
  index1 = 0
  index2 = len(arr) - 1
  while (index1 < index2):
    cur = arr[index1] + arr[index2]
    if cur == target:
      return True
    elif cur > target:
      index2 -= 1
    else:
      index1 += 1
  return False

arr = [-10, 1, 3, 5, 8, 10]
print(two_sum_sorted(arr, 0))
print(two_sum_sorted(arr, 1))
print(two_sum_sorted(arr, 4))
print(two_sum_sorted(arr, 18))
