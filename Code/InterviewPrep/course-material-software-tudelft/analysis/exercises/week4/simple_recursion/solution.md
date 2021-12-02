\\(n\\)  represents the number of elements which still need to be sorted, it corresponds to parameter \\(n\\).
Recurrence equation looks as follows:
\\(T(n)=T(n−1)+c_1n+c_2 \\)
and the corresponding base case equation
\\(T(0)=c_3\\)

- \\(c_3\\) correspond to the constant amount of time for performing the comparison.
- The overall loop takes \\(c_1n\\) time.
- Performing a comparison and assignments, takes constant time \\(c_2\\).
- Performing a recursive call with \\(n−1\\), takes \\(T(n−1)\\) time, since we decrease the size of n by 1 on each recursive call.

####Deriving the closed form:

$$
T(n)=T(n−1)+c_1n+c_2 \\\\
T(n)=(T(n−2)+c_1(n−1)+c_2)+c_1n+c_2=T(n−2)+c_1((n−1)+n)+2c_2 \\\\
T(n)=(T(n−3)+c_1(n−2)+c_2)+c_1((n−1)+n)+2c_2=T(n−3)+c_1((n−2)+(n−1)+n)+3c_2 \\\\
… \\\\
T(n)=T(n−n)+c_1(1+2+…+(n−1)+n)+c_2n \\\\
$$
####Simplification:
$$
T(n)=T(0)+c_1\sum_{i=1}^{n} i +c_2n \\\\
T(n)=c_1\sum_{i=1}^{n} i +c_2n+c_3 \\\\
T(n)=c_1\frac{n(n+1)}2+c_2n+c_3 \\\\
T(n)=c_1\frac{(n^2+n)}2+c_2n+c_3 \\\\
T(n)=\frac{c_1}2n^2+(\frac{c_1}2+c_2)n+c_3
$$

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n^{2} + n\\), which is a quadratic growth, since \\(n^{2}\\) grows faster than \\(n\\), on large inputs. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(n^{2})\\).