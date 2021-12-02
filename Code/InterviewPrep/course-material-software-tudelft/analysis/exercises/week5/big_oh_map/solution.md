1) Derive the run time equation of this code and explain all the terms.

We take \\(n\\) to be the number of nodes in the tree.

The algorithm combines a list of dictionaries and returns a single dictionary containing all entries.

\\(T(n)=\begin{cases}
    c_0 + m \cdot c_M & \text{if $n=1$}.\\\\
    c_1 + 2 \cdot T\left(\frac{n}{2}\right) & \text{otherwise}.
  \end{cases}\\)
  
Note that \\(c_M\\) represents the constant time operations required for adding one entry from one dictionary to another (line 14).
  
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​

\\(T(n) = c_1 + 2 \cdot T\left(\frac{n}{2}\right)\\)

We substitute \\(T\left(\frac{n}{2}\right)\\):

\\(T(n) = c_1 + 2 \cdot \left(c_1 + 2 \cdot T\left(\frac{n}{4}\right)\right)\\)
\\(T(n) = 3c_1 + 4 \cdot T\left(\frac{n}{4}\right)\\)

We substitute \\(T\left(\frac{n}{4}\right)\\):

\\(T(n) = 3c_1 + 4 \cdot \left(c_1 + 2 \cdot T\left(\frac{n}{8}\right)\right)\\)
\\(T(n) = 7c_1 + 8 \cdot T\left(\frac{n}{8}\right)\\)

For \\(k\\) repetitions of substitution:

\\(T(n) = (2^{k} - 1) \cdot c_1 + 2^{k} \cdot T\left(\frac{n}{2^{k}}\right)\\)

Take \\(k = \log n\\) to arrive at the base case:

\\(T(n) = (2^{\log n} - 1) \cdot c_1 + 2^{\log n} \cdot T\left(\frac{n}{2^{\log n}}\right)\\)

\\(T(n) = (n - 1) \cdot c_1 + n \cdot T\left(\frac{n}{n}\right)\\)

\\(T(n) = (n - 1) \cdot c_1 + n \cdot T\left(1\right)\\)

\\(T(n) = (n - 1) \cdot c_1 + n \cdot (c_0 + m \cdot c_M)\\)

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n \cdot m\\). Thus, the run time complexity is \\(\mathcal{O}(n \cdot m)\\).
