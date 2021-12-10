def longest_substr(s):
    chars = {}
    maxlen = 0
    left = 0
    for right in range(len(s)):
        if s[right] in chars:
            left = max(chars[s[right]] + 1, left)
        chars[s[right]] = right
        maxlen = max(maxlen, right - left + 1)
    return maxlen

print(longest_substr("abcabcdb")) # abcd (length 4)
print(longest_substr("bbbbb")) # b (length 1)
print(longest_substr("pwwkew")) # wke (length 3)
