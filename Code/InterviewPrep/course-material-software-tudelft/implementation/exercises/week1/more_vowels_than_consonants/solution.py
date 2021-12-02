from decorators import remove, empty


@empty
# Recursively determines if there are more vowels than consonants in the word.
# If there are more vowels than consonants, return 1
# If there are more consonants than vowels, return -1
# If there is an equal amount of vowels and consonants, return 0
def more_vowels(word: str) -> int:
    res = recursive_helper_linear(word, 0)
    # res = recursive_helper_quadratic(word)
    return 1 if res > 0 else 0 if res == 0 else -1


@remove
# We are slicing the string with `word[1:]`.
# This creates a new string with (n-1) characters and takes O(n) time.
def recursive_helper_quadratic(word: str):
    if len(word) == 0:
        return 0
    delta = 1 if word[0] in "aeiou" else -1
    return recursive_helper_quadratic(word[1:]) + delta


@remove
# Instead of slicing the word,
# we access the string directly at a specific index, which takes O(1) time.
def recursive_helper_linear(word: str, idx: int):
    if len(word) <= idx:
        return 0
    delta = 1 if word[idx] in "aeiou" else -1
    return recursive_helper_linear(word, idx + 1) + delta
