\\(n\\) represents the number of elements in the input list, it corresponds to `len(xs)`.
The recurrence equation looks as follows:
$$
T(n) = \begin{cases}
c_4 & \text{if } n < 2 \\\\
T(n-1)+c_1(n-1)+c_2l(n)+c_3 & \text{otherwise} 
\end{cases}
$$

- \\(c_4\\) corresponds to the constant amount of time for performing a comparison (line 5) and returning an empty list (line 6).
- \\(c_3\\) corresponds to the constant amount of time for performing a comparison (line 5), assigning two variables (lines 6-7) and returning from the function.
- The for-comprehension that creates `a` (line 6) takes \\(c_1(n-1)\\) time.
- The time needed for creating a list that contains all items of `a` and `b` depends on the amount of items in the lists. Adding each element takes constant time \\(c_2\\), and there are \\(l(n)\\) items in the list.
    - The amount of items in the resulting list is \\(l(n) = \begin{cases}
        0 & \text{if } n < 2 \\\\
        l(n-1) + (n-1) & \text{otherwise} 
        \end{cases}\\)
- Performing a recursive call with \\(n-1\\) (line 7), takes \\(T(n-1)\\) time, since we decrease the size of n by 1 on each recursive call.

####Deriving the closed form:

Let's first derive the closed form of \\(l(n)\\):
$$
\begin{align}
l(n)&=l(n-1)+(n-1) \\\\
l(n)&=l(n-2)+(n-2)+(n-1)=l(n-2)+(n-2)+(n-1) \\\\
l(n)&=l(n-3)+(n-3)+(n-2)+(n-1)=l(n-3)+(n-3)+(n-2)+(n-1) \\\\
    &\ldots \text{After $k$ iterations} \ldots \\\\
l(n)&=l(n-k)+(n-k)+(n-k+1)+\ldots+(n-2)+(n-1) \\\\
    &\text{ (take $n-k=1 \quad\to\quad k=n-1$)} \\\\
l(n)&=l(1)+1+2+\ldots+(n-2)+(n-1) \\\\
l(n)&=0+\sum_{i=1}^{n-1}i \\\\
l(n)&=\frac{n(n-1)}2
\end{align}
$$

We plug this into the recursive formula for \\(T(n)\\):
$$
T(n) = \begin{cases}
c_4 & \text{if } n < 2 \\\\
T(n-1)+c_1(n-1)+c_2\frac{n(n-1)}2+c_3 & \text{otherwise} 
\end{cases}
$$

Then the closed form for \\(T(n)\\) is:
$$
\begin{align}
T(n)&=T(n-1)+c_1(n-1)+c_2\frac{n(n-1)}2+c_3 \\\\
T(n)&=T(n-2)+c_1(n-2)+c_2\frac{(n-1)(n-2)}2+c_3 \quad+\quad c_1(n-1)+c_2\frac{n(n-1)}2+c_3 \\\\
T(n)&=T(n-2)+c_1(n-2+n-1)+c_2\frac{(n-1)(n-2)+n(n-1)}2+2c_3 \\\\
T(n)&=T(n-3)+c_1(n-3)+c_2\frac{(n-2)(n-3)}2+c_3 \quad+\quad c_1(n-2+n-1)+c_2\frac{(n-1)(n-2)+n(n-1)}2+2c_3 \\\\
T(n)&=T(n-3)+c_1(n-3+n-2+n-1)+c_2\frac{(n-2)(n-3)+(n-1)(n-2)+n(n-1)}2+3c_3 \\\\
    &\ldots \text{After $k$ iterations} \ldots \\\\
T(n)&=T(n-k)+c_1(n-k+\ldots+n-2+n-1)+c_2\frac{(n-k+1)(n-k)+\ldots+(n-1)(n-2)+n(n-1)}2+kc_3 \\\\
    &\text{ (take $n-k=1 \quad\to\quad k=n-1$)} \\\\
T(n)&=T(1)+c_1(1+\ldots+n-2+n-1)+c_2\frac{2\cdot1+\ldots+(n-1)(n-2)+n(n-1)}2+(n-1)c_3 \\\\
T(n)&=c_1\sum_{i=1}^{n-1}i + c_2\frac{\sum_{i=1}^{n-1}(i+1)i}2 + (n-1)c_3 + c_4 \\\\
T(n)&=c_1\frac{n(n-1)}2 + c_2\frac{\frac{(n-1)n(n+1)}3}2+(n-1)c_3 + c_4 \\\\
T(n)&=c_1\frac{n(n-1)}2 + c_2\frac{(n-1)n(n+1)}6+(n-1)c_3 + c_4 \\\\
\end{align}
$$
<sub>([Wolfram|Alpha: \\(\sum_{i=1}^{n-1} (i+1)i\\)](https://www.wolframalpha.com/input/?i=%5Csum_%7Bi%3D1%7D%5E%7Bn-1%7D+(i%2B1)i))</sub>

####Simplification:
$$
\begin{align}
T(n)&=c_1\frac{n(n-1)}2 + c_2\frac{(n-1)n(n+1)}6 + (n-1)c_3 + c_4 \\\\
T(n)&=c_1\frac{n^2 - n}2 + c_2\frac{\frac{n^3}3 - \frac{n}3}6 + nc_3 - c_3 + c_4 \\\\
T(n)&=c_1\frac{n^2}2-c_1\frac{n}2 + c_2\frac{n^3}{18}-c_2\frac{n}{18} + nc_3 - c_3 + c_4 \\\\
T(n)&=\frac{c_2}{18}n^3 + \frac{c_1}{2}n^2 + c_3n - \frac{c_1}2n - \frac{c_2}{18}n + c_4-c_3 \\\\
T(n)&=\frac{c_2}{18}n^3 + \frac{c_1}{2}n^2 + \frac{18c_3-9c_1-c_2}{18}n + c_4-c_3 \\\\
\end{align}
$$

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n^3 + n^2 + n\\), which is a cubic growth, since \\(n^3\\) grows faster than \\(n^2\\) and \\(n\\), on large inputs. Thus, the complexity in Big-Oh notation is \\(\mathcal{O}(n^3)\\).
