def longest_common_subsequence(text1, text2):
    n = len(text1)
    m = len(text2)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if text1[i] == text2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]
    
print(longest_common_subsequence("abcde", "acdef")) # 4 ("acde")
print(longest_common_subsequence("abc", "def")) # 0