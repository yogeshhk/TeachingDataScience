## Unfolding
We first derive the closed form by repeatedly unfolding.

$$
T(0) = c_0 \\\\
T(n) = T(n-2) + c_1 \\\\
$$

We fill in \\(T(n - 2)\\), where \\(T(n - 2) = T(n - 4) + c_1\\)
$$
T(n) = T(n - 4) + c_1 + c_1 \\\\
$$

We fill in \\(T(n - 4)\\), where \\(T(n - 4) = T(n - 6) + c_1\\)
$$
T(n) = T(n - 6) + c_1 + c_1 + c_1 \\\\
T(n) = T(n - 2 \cdot 3) + 3 \cdot c_1 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(n) = T(n - 2 \cdot k) + k \cdot c_1
$$

To substitute out the recursive part, \\(T(n - 2 \cdot k)\\), from the equation, we need to make \\(T(n - 2 \cdot k )\\) equal to the base case, \\(T(0)\\).
Which means that we need to solve the equation \\(n - 2 \cdot k = 0\\) for \\(k\\).

$$
n - 2 \cdot k = 0 \\\\
2k = n \\\\
k = \frac{1}{2} \cdot n \\\\
$$

So we can substitute \\(k\\) with \\(\frac{1}{2} \cdot n\\).

$$
T(n) = T(n - n) + \frac{1}{2} \cdot n \cdot c_1 \\\\
T(n) = T(0) + \frac{1}{2} \cdot n \cdot c_1 \\\\
T(n) = c_0 + \frac{1}{2} \cdot n \cdot c_1
$$

## Induction

To prove this we use proof by induction.

First we prove for the base case \\(T(0)\\).

$$
T(n) = c_0 + \frac{1}{2} \cdot n \cdot c_1 \\\\
T(0) = c_0 + \frac{1}{2} \cdot 0 \cdot c_1 \\\\
T(0) = c_0
$$

Next we have our induction hypothesis (IH).

$$
IH: T(n - 2) = c_0 + \frac{1}{2} \cdot (n - 2) \cdot c_1 
$$

We have to show that the recursive case \\(T(n) = T(n - 2) + c_1\\) correctly corresponds to the closed form solution.

$$
\begin{align}
T(n) &= T(n - 2) + c_1 \\\\
     &= _{IH} c_0 + \frac{1}{2} \cdot (n - 2) \cdot c_1 + c_1 \\\\
     &= c_0 + \frac{1}{2} \cdot n \cdot c_1
\end{align}
$$

By induction, it is now proven that \\(T(n) = c_0 + \frac{1}{2} \cdot n \cdot c_1\\) is correct for every integer \\(n \geq 0\\). \\(~ ~ \square\\)

## Big-Oh

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n\\), which is a linear growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(n)\\).
