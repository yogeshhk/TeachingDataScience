## Unfolding

$$
T(0, m) = m \cdot c_1 \\\\
T(h, m) = T(h - 1, m) + m \cdot c_2
$$

We fill in \\(T(h - 1, m)\\), where \\(T(h - 1, m) = T(h - 2, m) + m \cdot c_2\\)
$$
T(h, m) = T(h - 2, m) + m \cdot c_2 + m \cdot c_2 \\\\
T(h, m) = T(h - 2, m) + 2 \cdot m \cdot c_2
$$

We fill in \\(T(h - 2, m)\\), where \\(T(h - 2, m) = T(h - 3, m) + m \cdot c_2\\)
$$
T(h, m) = T(h - 3, m) + m \cdot c_2 + 2 \cdot m \cdot c_2 \\\\
T(h, m) = T(h - 3, m) + 3 \cdot m \cdot c_2 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(h, m) = T(h - k, m) + k \cdot m \cdot c_2 \\\\
$$

To substitute out the recursive part, \\(T(h - k, m)\\), from the equation, we need to make \\(T(h - k, m)\\) equal to the base case, \\(T(0, m)\\).
Which means that we need to solve the equation \\(h - k = 0\\) for \\(k\\).

$$
h - k = 0 \\\\
k = h
$$

So we can substitute \\(k\\) with \\(h\\).

$$
T(h, m) = T(h - h, m) + h \cdot m \cdot c_2 \\\\
T(h, m) = T(0, m) + h \cdot m \cdot c_2 \\\\
T(h, m) = m \cdot c_1 + h \cdot m \cdot c_2 \\\\
$$

## Induction

To prove this we use proof by induction.

First we prove for the base case \\(T(0, m)\\).

$$
T(h, m) = m \cdot c_1 + h \cdot m \cdot c_2 \\\\
T(0, m) = m \cdot c_1 + 0 \cdot m \cdot c_2 \\\\
T(0, m) = m \cdot c_1 \\\\
$$

Next we have our induction hypothesis (IH).

$$
IH: T(h - 1, m) = m \cdot c_1 + (h - 1) \cdot m \cdot c_2 \\\\
$$

We have to show that the recursive case T(h, m) = T(h - 1, m) + m \cdot c_2 correctly corresponds to the closed form solution.

$$
\begin{align}
T(h, m) &= T(h - 1, m) + m \cdot c_2 \\\\
     &= _{IH} m \cdot c_1 + (h - 1) \cdot m \cdot c_2 + m \cdot c_2 \\\\
     &= m \cdot c_1 + h \cdot m \cdot c_2 \\\\
\end{align}
$$

By induction, it is now proven that \\(T(h, m) = m \cdot c_1 + h \cdot m \cdot c_2\\) is correct for every integer \\(h \geq 0\\). \\(~ ~ \square\\)

## Big-Oh

After simplification, the constant terms and factors are removed. This leaves us with \\(T(h, m) = m + h \cdot m\\), where \\(h \cdot m\\) is the fastest growing factor. 
Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(h \cdot m)\\).
