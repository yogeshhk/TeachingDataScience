# Return the longest common prefix
def longest_common_prefix(strs):
    cur_index = 0
    prefix = ""
    for cur_char in strs[0]:
        for i in range(1, len(strs)):
            if (cur_index >= len(strs[i]) or 
                    strs[i][cur_index] != cur_char):
                return prefix
        prefix += cur_char
        cur_index += 1
    return prefix

print(longest_common_prefix(["plank","planter","planking"]))
print(longest_common_prefix(["hello","foo","bar"]))
