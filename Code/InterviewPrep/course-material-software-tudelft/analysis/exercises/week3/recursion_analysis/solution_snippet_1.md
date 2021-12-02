\\(T(n)=\begin{cases}
    c_0 & \text{if $n=0$}.\\\\
    T(n-1) + c_1 & \text{otherwise}.
  \end{cases}\\)
  
Where:
* \\(n\\) is the value of the integer `n`
* \\(c_0\\) represents the constant-time operations on lines 2 and 3, when `n == 0`
* \\(c_1\\) represents the constant-time operations needed for checking the condition and the computations in line 4

By repeatedly unfolding we can find the closed form:

We fill in \\(T(n - 1)\\), where \\(T(n - 1) = T(n - 2) + c_1\\)
$$
T(n) = T(n - 2) + c_1 + c_1 \\\\
$$

We fill in \\(T(n - 2)\\), where \\(T(n - 2) = T(n - 3) + c_1\\)
$$
T(n) = T(n - 3) + c_1 + c_1 + c_1 \\\\
T(n) = T(n - 3) + 3 \cdot c_1 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(n) = T(n - k) + k \cdot c_1
$$

To substitute out the recursive part, \\(T(n - k)\\), from the equation, we need to make \\(T(n - k )\\) equal to the base case, \\(T(0)\\).
Which means that we need to solve the equation \\(n - k = 0\\) for \\(k\\).

$$
n - k = 0 \\\\
k = n \\\\
$$

So we can substitute \\(k\\) with \\(n\\).

$$
T(n) = T(n - n) + n \cdot c_1 \\\\
T(n) = T(0) + n \cdot c_1 \\\\
T(n) = c_0 + n \cdot c_1
$$

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n\\), which is a linear growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(n)\\).