## Unfolding

$$
T(1) = c_0 \\\\
T(n) = T(\frac{n}{2}) + c_1 \\\\
$$

We fill in \\(T(\frac{n}{2})\\), where \\(T(\frac{n}{2}) = T(\frac{n}{4}) + c_1\\)
$$
T(n) = T(\frac{n}{4}) + c_1 + c_1 \\\\
$$

We fill in \\(T(\frac{n}{4})\\), where \\(T(\frac{n}{4}) = T(\frac{n}{8}) + c_1\\)
$$
T(n) = T(\frac{n}{8}) + c_1 + c_1 + c_1 \\\\
T(n) = T(\frac{n}{2^3}) + 3 c_1 \\\\
$$

By repeating this \\(k\\) times we get

$$
T(n) = T(\frac{n}{2^k}) + k c_1
$$

To substitute out the recursive part, \\(T(\frac{n}{2^k})\\), from the equation, we need to make \\(T(\frac{n}{2^k})\\) equal to the base case, \\(T(1)\\).
Which means that we need to solve the equation \\(\frac{n}{2^k} = 1\\) for \\(k\\).

$$
\frac{n}{2^k} = 1 \\\\
n = 2^k \\\\
k = log(n) \\\\
$$

So we can substitute \\(k\\) with \\(log (n)\\).

$$
T(n) = T(\frac{n}{n}) + c_1 \cdot log(n) \\\\
T(n) = T(1) + c_1 \cdot log(n) \\\\
T(n) = c_0 + c_1 \cdot log(n) \\\\
$$

## Induction

To prove this we use proof by induction.

First we prove for the base case \\(T(0)\\).

$$
T(n) = c_0 + c_1 \cdot log(n) \\\\
T(1) = c_0 + c_1 \cdot log(1) \\\\
T(1) = c_0
$$

Next we have our induction hypothesis (IH).

$$
\begin{align}
IH: T(\frac{n}{2}) &= c_0 + c_1 \cdot log(\frac{n}{2}) \\\\
                   &= c_0 + c_1 \cdot (log(n) - 1)
\end{align}
$$

We have to show that the recursive case \\(T(n) = T(\frac{n}{2}) + c_1\\) correctly corresponds to the closed form solution.

$$
\begin{align}
T(n) &= T(\frac{n}{2}) + c_1 \\\\
     &= _{IH} c_0 + c_1 \cdot (log(n) - 1) + c_1 \\\\
     &= c_0 + c_1 \cdot log(n) - c_1 + c_1 \\\\
	 &= c_0 + c_1 \cdot log(n) \\\\
\end{align}
$$

By induction, it is now proven that \\(T(n) = c_0 + c_1 \cdot log(n)\\) is correct for every integer \\(n \geq 1\\). \\(~ ~ \square\\)

## Big-Oh

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = log(n)\\), which is a logarithmic growth. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(log(n))\\).
