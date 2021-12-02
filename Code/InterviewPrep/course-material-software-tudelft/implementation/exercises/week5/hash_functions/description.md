In this exercise, you will implement several well-known hash functions for strings.

The hash function will be defined mathematically.
\\(b_0\\) is the starting value and \\(b_i\\) are the intermediate values **of the hash code** for the string up to character \\(i\\).
\\(s\\) represents the size of the hash table, which is given as a parameter to your function.
\\(n\\) represents the length of the string.
\\(c_i\\) is the (integer) value of the \\(i^\mathrm{th}\\) character of the string, where \\(1 \leq i \leq n\\) (the first character is at \\(c_1\\), unlike in Python).
The final value is given as \\(H\\).

You should implement the following string hashing algorithms:

#### A variant on the GNU-cc1 hashing algorithm:
- \\(b_0 = n\\)
- \\(b_i = 33b_{i-1} \text{ xor } c_i \\)
- \\(H = b_n \mod s\\)

#### A variant on the GNU-cpp hashing algorithm:
- \\(b_0 = 0\\)
- \\(b_i = 4b_{i-1} + c_i \\)
- \\(H = b_n \mod s\\)

#### Python's `hash` function
For this variant, you can use Python's built-in `hash` function.
Make sure that the hash value as returned by `hash` is in the correct range!
If the hash value is negative, make it positive by flipping the sign.

<details>
<summary>Hint: the xor operator</summary>
In Python, "c = a xor b" is written as <code>c = a ^ b</code>.
</details>

<details>
<summary>Hint: getting the value of a string character in Python</summary>
To get the value of a string character, you can use Python's built-in <code>ord</code> function.

For example: <code>ord("a") = 97</code>
</details>
