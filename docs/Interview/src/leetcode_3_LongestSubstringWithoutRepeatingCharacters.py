#  Given a string find the length of the longest substring without repeating characters
#   Example: "abcabcbb" => 3 for "abc"
#            "bbbbbbbbbb" => 1 for "b"
# Ref:https://www.youtube.com/watch?v=sUicrnHwA0s&list=PLiC1doDIe9rDFw1v-pPMBYvD6k1ZotNRO&index=3

# Proceed till we hit duplicate character, store it
# Start from next of the conflict duplicate character
def lengthOfLongestSubstring(s):
    substring = dict()
    current_substring_start = 0
    current_substring_length = 0
    longest_substring_sofar = 0
    for i, letter in enumerate(s):
        if letter in substring and substring[letter] >= current_substring_start:
            current_substring_start = substring[letter] + 1
            current_substring_length = i - substring[letter]
            substring[letter] = i
        else:
            substring[letter] = i
            current_substring_length += 1
            if current_substring_length > longest_substring_sofar:
                longest_substring_sofar = current_substring_length
    return longest_substring_sofar
