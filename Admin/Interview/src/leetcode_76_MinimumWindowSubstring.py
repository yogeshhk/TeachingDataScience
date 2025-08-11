# Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every
# character in t (including duplicates) is included in the window. If there is no such substring, return the empty
# string "".
#
# The testcases will be generated such that the answer is unique.
#
# A substring is a contiguous sequence of characters within the string.
#
#
#
# Example 1:
#
# Input: s = "ADOBECODEBANC", t = "ABC"
# Output: "BANC"
# Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

# Brute Force: Make dict of 't' and of all possible windows as well and then find subset which qualify then min of them

def minimum_window(s, t):
    if t == "": return ""
    t_dict = {}
    window_dict = {}
    for c in t:
        t_dict[c] = 1 + t_dict.get(c, 0)

    have_count = 0
    need_count = len(t_dict)
    result = [-1,-1] # range indices
    result_length = float("infinity") # some big number now
    left = 0
    for right in range(len(s)):
        c = s[right]
        window_dict[c] = 1 + window_dict.get(c, 0)
        if c in t_dict and window_dict[c] == t_dict[c]:
            have_count += 1
        while have_count == need_count:
            if (right - left + 1) < result_length:
                result = [left,right]
                result_length = right - left + 1
            window_dict[s[left]] -= 1
            if s[left] in t_dict and window_dict[s[left]] < t_dict[s[left]]:
                have_count -= 1
            left += 1
    left, right = result
    return s[left: right+1] if result_length != float("infinity") else ""

s = "ADOBECODEBANC"
t = "ABC"
result = minimum_window(s,t)
print(result)