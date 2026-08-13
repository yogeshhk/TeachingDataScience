# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:

def guessNumber(n):
    left, right = 1, n
    while True:
        mid = (left + right)//2
        user_response = guess(mid)
        if user_response > 0:
            left = mid + 1
        elif user_response < 0 :
            right = mid - 1
        else:
            return mid