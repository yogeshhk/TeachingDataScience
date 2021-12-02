The Factorial Poster Company (FPC) prints posters that display the result of
any factorial that their customers wish for.
Recently, they got an order from Professor D.R. Ingenious, who wants to do an
experiment with very large factorial numbers.
The FPC want to be as efficient with printing as possible, and therefore they
decided on a way to save paper.
Since the larger factorial numbers end in a lot of zeroes, they decide to cut
off this number of zeroes \\(z\\) and replace it with "\\(\cdot 10^z\\)".

You are given the task to calculate, for every order of Prof. Ingenious, how
many zeroes \\(z\\) can be cut off from the poster, so that the FPC know how
much poster paper they will save.

#### Input
One line containing one integer \\(n\\), with \\(0 \leq n \leq 10^{18}\\).

#### Output
One line containing one integer \\(z\\), the amount of trailing zeroes of \\(n!\\).
Note that any other zeroes in the result of \\(n!\\) do not count, see the second
example.

#### Examples
For each example, the first block is the input and the second block is the corresponding output.
##### Example 1
```
1
```
```
0
```
##### Example 2
```
7
```
```
1
```
##### Example 3
```
42
```
```
9
```

Source: FPC 2018
