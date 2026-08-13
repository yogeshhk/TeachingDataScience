# https://www.geeksforgeeks.org/generate-all-binary-strings-from-given-pattern/
# Given a string containing of ‘0’, ‘1’ and ‘?’ wildcard characters,
# generate all binary strings that can be formed by replacing each wildcard character by ‘0’ or ‘1’.

def generate_binaries_recursion(string_list, i):
    if i == len(string_list):
        print("".join(string_list))
        return
    if string_list[i] == "?":
        string_list[i] = '0'
        generate_binaries_recursion(string_list, i + 1)
        string_list[i] = '1'
        generate_binaries_recursion(string_list, i + 1)
        string_list[i] = "?"
    else:
        generate_binaries_recursion(string_list, i + 1)


def generate_binaries_queue(string):
    string_queue = [string]
    while len(string_queue) > 0:
        current_string = string_queue[0]
        if "?" in current_string:
            s1 = current_string.replace("?","0",1)
            string_queue.append(s1)
            s2 = current_string.replace("?","1",1)
            string_queue.append(s2)
        else: # no wildcards left
            print(current_string)
        string_queue.pop(0)






pattern = "1??0?101"
pattern_list= list(pattern)
# generate_binaries_recursion(pattern_list, 0)
generate_binaries_queue(pattern)

