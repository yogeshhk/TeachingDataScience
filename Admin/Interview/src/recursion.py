
def fibonacci_recursion(n):
    if n < 0:
        print("Error")
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci_recursion(n-1) + fibonacci_recursion(n-2)

fibonacci_array = [1,1]
def fibonacci_dynamic(n):
    if n < 0:
        print("Error")
    elif n <= len(fibonacci_array):
        return fibonacci_array[n-1]
    else:
        next_num = fibonacci_dynamic(n-1) + fibonacci_dynamic(n-2)
        fibonacci_array.append(next_num)
        return next_num

print(fibonacci_recursion(9))
print(fibonacci_dynamic(9))

