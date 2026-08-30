
def linear_search(array,number):
    position = 0
    while True:
        if array[position] == number:
            return position
        position += 1
        if position == len(array):
            return -1

def binary_search(array, number):
    low = 0
    high = len(array) - 1
    while low <= high:
        mid = (low + high) // 2
        if number == array[mid]:
            return mid
        elif number < array[mid]:
            high = mid - 1
        elif number > array[mid]:
            low = mid + 1
    return -1

tests = []
test = {
    'input': {
        'array' : [10, 2, 4, 16, 37, 3],
        'number' : 5
    },
    'output': -1
}
tests.append(test)

test = {
    'input': {
        'array' : [10, 2, 4, 16, 37, 3],
        'number' : 4
    },
    'output': 2
}
tests.append(test)

for test in tests:
    print(binary_search(**test['input']) == test['output'])
