# Given array of digits of an integer, add one to it
# Example: [1,2,3] => [1,2,4]

def plusOne_string(digits):
    num_string = "".join([str(i) for i in digits])
    num_int = int(num_string)
    plus_int = num_int + 1
    return [int(c) for c in str(plus_int)]

def plusOne_carryforward(digits):
    for i in range(len(digits)-1,-1,-1):
        if digits[i] == 9:
            digits[i] = 0
        else:
            digits[i] += 1
            return digits
    return [1] + digits

