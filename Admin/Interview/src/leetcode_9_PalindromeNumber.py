# Determine if given integer is palindrome, without converting integer to string
# Example: 121 is, -121 is not

def is_palindrome(x):
    if x < 0:
        return False
    reversed_num = 0
    decimal_step = 0
    while x // (10 ** decimal_step) !=0:
        reversed_num = (reversed_num * 10) + (x // (10 ** decimal_step) % 10)
        decimal_step += 1

    return x == reversed_num

