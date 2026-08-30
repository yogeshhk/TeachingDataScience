# https://www.geeksforgeeks.org/find-the-longest-substring-with-k-unique-characters-in-a-given-string/
# Given a string you need to print longest possible substring that has exactly M unique characters.
# If there are more than one substring of the longest possible length, then print any one of them.
#
# Examples:
#
# "aabbcc", k = 1
# Max substring can be any one from {"aa" , "bb" , "cc"}.
#
# "aabbcc", k = 2
# Max substring can be any one from {"aabb" , "bbcc"}.

def find_longest_substring_bruteforce(s, k):
    results = []
    max_len = -1
    for i in range(len(s)+1):
        for j in range(len(s)+1):
            substring = s[i:j]
            unique_chars = len(set(list(substring)))
            if unique_chars == k:
                if len(substring) >= max_len:
                    max_len = len(substring)
                results.append(substring)
    results = [r for r in results if len(r) == max_len]
    return results


print(find_longest_substring_bruteforce("aaabbb", 3))
