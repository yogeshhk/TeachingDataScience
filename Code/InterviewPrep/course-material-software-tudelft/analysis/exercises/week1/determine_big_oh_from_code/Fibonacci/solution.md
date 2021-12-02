\\(T(n)=\begin{cases}
    c_0 & \text{if $n=0$ or $n=1$}.\\\\
    c_1 + T(n-1) + T(n-2) & \text{otherwise}.
  \end{cases}\\)

Where:
* \\(n\\) is the value of `i`
* \\(c_0\\) represents the constant-time operations in lines 9-12
* \\(c_1\\) represents the constant-time operations in lines 9, 11 and 13

We replace \\(T(n-2)\\) by \\(T(n-1)\\), as \\(T(n-2)\\) has an upper bound of \\(T(n-1)\\). 
This simplification means our time complexity no longer gives the tightest bound, but that was not required.

\\(T(n) = c_1 + 2 \cdot T(n-1)\\)

\\(T(n) = c_1 + 2 \cdot (c_1 + 2 \cdot T(n-2))\\)

\\(T(n) = 3 \cdot c_1 + 4 \cdot T(n-2))\\)

We fill in \\(T(n-2)\\)

\\(T(n) = 7 \cdot c_1 + 8 \cdot T(n-3))\\)

We fill in \\(T(n-3)\\)

\\(T(n) = 15 \cdot c_1 + 16 \cdot T(n-4))\\)

For \\(k\\) repetitions of subtitution:

\\(T(n) = (2^k - 1) \cdot c_1 + 2^k \cdot T(n-k))\\)

Let \\(k = n\\) to arrive at the base case:

\\(T(n) = (2^n - 1) \cdot c_1 + 2^n \cdot T(n-n))\\)

\\(T(n) = (2^n - 1) \cdot c_1 + 2^n \cdot T(0))\\)

\\(T(n) = (2^n - 1) \cdot c_1 + 2^n \cdot c_0\\)

After simplification, the constant terms and factors are removed.
This leaves us with \\(T(n)=2^n\\), which is a exponential growth.
Thus, the computational complexity is \\(O(2^n)\\).
