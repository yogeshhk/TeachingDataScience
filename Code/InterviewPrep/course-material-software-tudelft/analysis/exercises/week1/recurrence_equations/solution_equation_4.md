## Unfolding
We first derive the closed form by repeatedly unfolding.

$$
T(1) = c_0 \\\\
T(n) = 2 \cdot T(n - 1) + c_1 \\\\
$$

We fill in \\(T(n - 1)\\), where \\(T(n - 1) = 2 \cdot T(n - 2) + c_1\\)
$$
T(n) = 2 \cdot (2 \cdot T(n - 2) + c_1) + c_1 \\\\
T(n) = 2^{2} \cdot T(n - 2) + 3 \cdot c_1 \\\\
$$

We fill in \\(T(n - 2)\\), where \\(T(n - 2) = 2 \cdot T(n - 3) + c_1\\)
$$
T(n) = 2^{2} \cdot ( 2 \cdot T(n - 3) + c_1) + 3 \cdot c_1 \\\\
T(n) = 2^{3} \cdot T(n - 3) + 7 \cdot c_1 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(n) = 2^k \cdot T(n - k) + (2^k - 1) \cdot c_1
$$

To substitute out the recursive part, \\(T(n - k)\\), from the equation, we need to make \\(T(n - k )\\) equal to the base case, \\(T(1)\\).
Which means that we need to solve the equation \\(n - k = 0\\) for \\(k\\).

$$
n - k = 1 \\\\
k = n - 1 \\\\
$$

So we can substitute \\(k\\) with \\((n - 1)\\).

$$
T(n) = 2^{n - 1} \cdot T(n - (n - 1)) + (2^{n - 1} - 1) \cdot c_1 \\\\
T(n) = 2^{n - 1} \cdot T(1) + (2^{n - 1} - 1) \cdot c_1 \\\\
T(n) = 2^{n - 1} \cdot c_0 + (2^{n - 1} - 1) \cdot c_1
$$

## Induction

To prove this we use proof by induction.

First we prove for the base case \\(T(1)\\).

$$
T(n) = 2^{n - 1} \cdot c_0 + (2^{n - 1} - 1) \cdot c_1 \\\\
T(1) = 2^{1 - 1} \cdot c_0 + (2^{1 - 1} - 1) \cdot c_1 \\\\
T(1) = 2^{0} \cdot c_0 + (2^{0} - 1) \cdot c_1 \\\\
T(1) = c_0
$$

Next we have our induction hypothesis (IH).

$$
IH: T(n - 1) = 2^{n - 2} \cdot c_0 + (2^{n - 2} - 1) \cdot c_1
$$

We have to show that the recursive case \\(T(n) = 2 \cdot T(n - 1) + c_1\\) correctly corresponds to the closed form solution.

$$
\begin{align}
T(n) &= 2 \cdot T(n - 1) + c_1 \\\\
     &= _{IH} 2 \cdot (2^{n - 2} \cdot c_0 + (2^{n - 2} - 1) \cdot c_1) + c_1 \\\\
     &= 2^{n - 1} \cdot c_0 + 2^{n - 1} \cdot c_1 - c_1 \\\\
     &= 2^{n - 1} \cdot c_0 + (2^{n - 1} - 1) \cdot c_1
\end{align}
$$

By induction, it is now proven that \\(T(n) = 2^{n - 1} \cdot c_0 + (2^{n - 1} - 1) \cdot c_1\\) is correct for every integer \\(n \geq 1\\). \\(~ ~ \square\\)

## Big-Oh

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = 2^n\\), which is an exponential growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(2^n)\\).
