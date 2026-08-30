# The DNA sequence is composed of a series of nucleotides abbreviated as 'A', 'C', 'G', and 'T'.
#
# For example, "ACGAATTCCG" is a DNA sequence.
# When studying DNA, it is useful to identify repeated sequences within the DNA.
#
# Given a string s that represents a DNA sequence, return all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.
#
#
#
# Example 1:
#
# Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
# Output: ["AAAAACCCCC","CCCCCAAAAA"]

# Brute force, collect all 10 long string and make counters dictionary

def find_repeated_DNA_sequence(s):
    seen = set()
    result = set()

    for left in range(len(s) - 9):
        current = s[left : left + 10]
        if current in seen:
            result.add(current)
        seen.add(current)
    return list(result)

s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
result = find_repeated_DNA_sequence(s)
print(result)
