# Given a string s, return longest palindromic substring in s
# Example: "babad" => "bab", "aba"

# Ref:https://www.youtube.com/watch?v=sUicrnHwA0s&list=PLiC1doDIe9rDFw1v-pPMBYvD6k1ZotNRO&index=4

# Brute force could be to run 2 for loops, for different ranges, and check for palindrome, return longest

def is_palindrome(s):
    return s == s[::-1]


def longestPalindromicSubstring_bruteforce(s):
    for length in range(len(s), 0, -1):  # reverse length numbers
        for start_index in range(0, len(s) + 1 - length):  # different starting points
            current_substring = s[start_index:(start_index + length)]
            if is_palindrome(current_substring):
                return current_substring  # return right away because we are checking the longest first


def longestPalindromicSubstring(s):
    biggest = s[0]
    step = len(biggest) // 2  # one side
    # odd case
    for center in range(1, len(s) - 1):
        bounds = [center - (1 + step), center + (1 + step)]  # both directions
        while (bounds[0] > -1) and (bounds[1] < len(s)):  # ends
            current_string = s[bounds[0]:bounds[1] + 1]
            if is_palindrome(current_string):
                biggest = current_string
                step = len(biggest) // 2  # find longer next
                bounds[0] -= 1  # make wider
                bounds[1] += 1
            else:
                break

    # even case
    for center in range(step, len(s) - step - 1):
        bounds = [center - step, center + (1 + step)]  # both directions
        while (bounds[0] > -1) and (bounds[1] < len(s)):  # ends
            current_string = s[bounds[0]:bounds[1] + 1]
            if is_palindrome(current_string):
                biggest = current_string
                step = len(biggest) // 2  # find longer next
                bounds[0] -= 1  # make wider
                bounds[1] += 1
            else:
                break
    return biggest