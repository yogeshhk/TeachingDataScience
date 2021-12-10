# Return true if the string is a palindrome, and false otherwise
def is_palin(s):
    start = 0
    end = len(s) - 1

    while (start < end):
        if (s[start] != s[end]):
            return False
        start += 1
        end -= 1
    return True

print(is_palin("racecar"))
print(is_palin("abcddcba"))
print(is_palin("hello"))
print(is_palin("foobarfoobar"))