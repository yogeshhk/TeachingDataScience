# https://www.geeksforgeeks.org/count-strings-can-formed-using-b-c-given-constraints/
# Given a length n, count the number of strings of length n that can be made using ‘a’, ‘b’ and ‘c’
# with at-most one ‘b’ and two ‘c’s allowed.

def count_strings(n, bcount, ccount):
    if bcount < 0 or ccount < 0:
        return 0
    if n == 0:
        return 1
    if bcount == 0 and ccount == 0:
        return 1

    results = count_strings(n-1,bcount,ccount)
    results += count_strings(n - 1, bcount-1, ccount)
    results += count_strings(n -1, bcount, ccount-1)

    return results

print(count_strings(3,1,2))


