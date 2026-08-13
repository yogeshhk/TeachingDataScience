# https://www.geeksforgeeks.org/find-largest-word-dictionary-deleting-characters-given-string/
# Giving a dictionary and a string ‘str’, find the longest string in dictionary
# which can be formed by deleting some characters of the given ‘str’.

def find_largest_word(words, word):
    word_count = {}
    for c in word:
        if c in word_count:
            word_count[c] += 1
        else:
            word_count[c] = 1
    max_word = ""
    for w in words:
        word_count_copy = word_count.copy()
        w_len = len(w)
        for c in w:
            if c in word_count_copy and word_count_copy[c] != 0:
                word_count_copy[c] -= 1
            else:
                w_len -= 1
        if w_len > len(max_word):
            max_word = w
    return max_word


def is_subsequence(str1, str2):
    m = len(str1)
    n = len(str2)
    i, j = 0, 0
    while (i < m and j < n):
        if str1[i] == str2[j]:
            j += 1
        i += 1
    return (i == m)


def find_largest_word_subsequence(words, word):
    max_word = ""
    for w in words:
        if is_subsequence(w, word):
            if len(w) > len(max_word):
                max_word = w
    return max_word


# words = ["ale", "apple", "monkey", "plea"]
words = ["pintu", "geeksfor", "geeksgeeks", " forgeek"]
word = "geeksforgeeks"
# word = "abpcplea"

print(find_largest_word_subsequence(words, word))
