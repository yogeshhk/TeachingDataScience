# https://www.geeksforgeeks.org/meta-strings-check-two-strings-can-become-swap-one-string/
# Given two strings, the task is to check whether these strings are meta strings or not.
# Meta strings are the strings which can be made equal by exactly one swap in any of the strings.
# Equal string are not considered here as Meta strings.
# Examples:
#
# Input: str1 = "geeks"
# str2 = "keegs"
# Output: Yes
# By just swapping 'k' and 'g' in any of string, both will become same.
#
# Input: str1 = "rsting"
# str2 = "string
# Output: No

def is_meta_string(str1, str2):
    if len(str1) != len(str2):
        return False
    mismatches = 0
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            mismatches += 1
    return mismatches == 2

print(is_meta_string("geeks","keegs"))
print(is_meta_string("rsting","string"))