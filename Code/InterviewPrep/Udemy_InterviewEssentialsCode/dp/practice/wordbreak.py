def word_break(s, word_dict):
    words = set(word_dict)
    dp = [False]*(len(s) + 1)
    dp[0] = True
        
    for i in range(len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break

    return dp[-1]
        
print(word_break("applepenapple", ["apple", "pen"])) # True ("apple pen apple")
print(word_break("catsandog", ["cats", "and", "dog", "sand", "cat"])) # False
