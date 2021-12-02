## Unfolding

$$
T(0) = c_1 \\\\
T(g) = 3 \cdot T(g - 1) + c_2 \\\\
$$

We fill in \\(T(g - 1)\\), where \\(T(g - 1) = 3 \cdot T(g - 2) + c_2\\)
$$
T(g) = 3 \cdot (3 \cdot T(g - 2) + c_2) + c_2 \\\\
T(g) = 3^{2} \cdot T(g - 2) + (3 + 1) \cdot c_2
$$

We fill in \\(T(g - 2)\\), where \\(T(g - 2) = 3 \cdot T(g - 3) + c_2\\)
$$
T(g) = 3^{2} \cdot (3 \cdot T(g - 3) + c_2) + (3 + 1) \cdot c_2\\\\
T(g) = 3^{3} \cdot T(g - 3) + (3^{2} + 3 + 1) \cdot c_2 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(g) = 3^{k} \cdot T(g - k) + (3^{k - 1} + ... + 3 + 1) \cdot c_2 \\\\
T(g) = 3^{k} \cdot T(g - k) + \left(\frac{3^{k} - 1}{2}\right) \cdot c_2 \\\\
$$

To substitute out the recursive part, \\(T(g - k)\\), from the equation, we need to make \\(T(g - k)\\) equal to the base case, \\(T(0)\\).
Which means that we need to solve the equation \\(g - k = 0\\) for \\(k\\).

$$
g - k = 0 \\\\
k = g
$$

So we can substitute \\(k\\) with \\(g\\).

$$
T(g) = 3^{g} \cdot T(g - g) + \left(\frac{3^{g} - 1}{2}\right) \cdot c_2 \\\\
T(g) = 3^{g} \cdot T(0) + \left(\frac{3^{g} - 1}{2}\right) \cdot c_2 \\\\
T(g) = 3^{g} \cdot c_1 + \left(\frac{3^{g} - 1}{2}\right) \cdot c_2 \\\\
T(g) = 3^{g} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{c_2}{2} \\\\
$$

## Induction

To prove this we use proof by induction.

First we prove for the base case \\(T(0)\\).

$$
T(g) = 3^{g} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{c_2}{2} \\\\
T(0) = 3^{0} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{c_2}{2} \\\\
T(0) = c_1
$$

Next we have our induction hypothesis (IH).

$$
IH: T(g - 1) = 3^{g - 1} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{c_2}{2} \\\\
$$

We have to show that the recursive case \\(T(g) = 3 \cdot T(g - 1) + c_2\\) correctly corresponds to the closed form solution.

$$
\begin{align}
T(g) &= 3 \cdot T(g - 1) + c_2 \\\\
     &= _{IH} 3 \cdot \left(3^{g - 1} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{c_2}{2}\right) + c_2 \\\\
     &= 3^{g} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{3c_2}{2} + c_2 \\\\
     &= 3^{g} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{c_2}{2} \\\\
\end{align}
$$

By induction, it is now proven that \\(T(g) = 3^{g} \cdot \left(c_1 + \frac{c_2}{2}\right) - \frac{c_2}{2}\\) is correct for every integer \\(n \geq 0\\). \\(~ ~ \square\\)

## Big-Oh

After simplification, the constant terms and factors are removed. This leaves us with \\(T(g) = 3^{g}\\), which is an exponential growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(3^{g})\\).
