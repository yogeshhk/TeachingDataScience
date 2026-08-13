# Find length of last word from a sentence
# Example: For "Hello World" = 5

# String split
def lengthOfLastWord_split(s):
    words = s.split()
    if words:
        return len(words[-1])
    return 0

# Manual check
def lengthOfLastWord_spaces(s):
    count = 0
    for letter in s[::-1]:
        if letter == " ":
            if count >= 1:
                return count
        else:
            count += 1
    return count
