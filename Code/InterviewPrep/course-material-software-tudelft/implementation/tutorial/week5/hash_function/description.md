In this exercise, you will implement a hash function for strings.

The hash function will be defined mathematically.
\\(b_0\\) is the starting value and \\(b_i\\) are the intermediate values **of the hash code** for the string up to character \\(i\\).
\\(s\\) represents the size of the hash table, which is given as a parameter to your function.
\\(n\\) represents the length of the string.
\\(c_i\\) is the (integer) value of the \\(i^\mathrm{th}\\) character of the string, where \\(1 \leq i \leq n\\) (the first character is at \\(c_1\\), unlike in Python).
The final value is given as \\(H\\).

You should implement the following string hashing algorithm:

- \\(a = 1003\\)
- \\(m = 10^9 + 9\\)
- \\(b_0 = 0\\)
- \\(b_i = c_i + b_{i-1} \cdot a \mod m\\)
- \\(H = b_n \mod s\\)
