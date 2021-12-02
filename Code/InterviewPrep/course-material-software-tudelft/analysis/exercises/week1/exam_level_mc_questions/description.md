Question 1:
Consider four functions $f(n), g(n), h(n)$ and $z(n)$ such that:

 - \\(f(n)\\) is \\(O(g(n))\\).
 - \\(g(n)\\) is \\(\Theta(h(n))\\).
 - \\(h(n)\\) is \\(\Omega(z(n))\\).

Which of the following can we now conclude with certainty?

[] \\(f(n)\\) is \\(\Omega(h(n))\\)
[] \\(g(n)\\) is \\(\Theta(z(n))\\)
[] \\(h(n)\\) is \\(\Theta(f(n))\\)
[x] \\(z(n)\\) is \\(O(g(n))\\)

Solution 1:
A. Take \\(f(n) = 1\\), \\(g(n) = n^2\\), \\(h(n) = n^2\\), \\(z(n) = n\\). This makes all conditions true, but not answer A.
B. Take \\(f(n) = 1\\), \\(g(n) = n^2\\), \\(h(n) = n^2\\), \\(z(n) = n\\). This makes all conditions true, but not answer B.
C. Take \\(f(n) = 1\\), \\(g(n) = n^2\\), \\(h(n) = n^2\\), \\(z(n) = n\\). This makes all conditions true, but not answer C.
D. From point 3 we can derive \\(z(n)\\) is \\(O(h(n))\\) and combine this with point 2 and we know that \\(z(n)\\) is also \\(O(g(n))\\).

Question 2:
Which of the following statements about asymptotic complexity is **true**?
[] If \\(f(n)\\) is \\(O(g(n))\\) and \\(g(n)\\) is \\(\Omega(f(n))\\) then \\(f(n)\\) is \\(\Theta(g(n))\\).
[] If \\(f(n)\\) is \\(\Omega(g(n))\\) and \\(g(n)\\) is \\(O(f(n))\\) then \\(f(n)\\) is \\(\Theta(g(n))\\).
[x] If \\(f(n)\\) is \\(O(g(n))\\) and \\(g(n)\\) is \\(O(h(n))\\) then \\(f(n)\\) is \\(O(h(n))\\).
[] If \\(f(n)\\) is \\(O(g(n))\\) and \\(g(n)\\) is \\(\Omega(h(n))\\) then \\(f(n)\\) is \\(\Omega(h(n))\\).

Solution 2:
A. False, take \\(f(n) = n\\), \\(g(n) = n^2\\).
B. Same as \\(A\\), just flip \\(f\\) and \\(g\\).
C. True, it means \\(f(n) \geq c_g g(n) \geq c_g c_h h(n)\\) for some \\(n \geq \max(n_g,n_h)\\).
D. False, take \\(f(n) = n\\), \\(g(n) = n^2\\), and \\(h(n) = n \log n\\).

Question 3:
Which of the following statements is **false**?
[]  \\(42000000\\) is \\(O(n^{42})\\).
[x]  \\(8n^2 + 15\\) is \\(O(n\log n)\\).
[] \\(7n^2 + 3n + 2\\) is \\(\Theta(n^2)\\).
[]  \\(18n^3 + 42n \log n + \sqrt{n}\\) is \\(\Omega(n)\\)

Solution 3:
All are true, except B. \\(8n^2\\) is \\(\Omega(n^2)\\) and so the whole expression cannot be \\(O(n\log n)\\).	

Question 4:
Consider the following code snippet:
``` python
def methodMC2(n : int) -> int:
    if n == 0:
        return 1
    a = 0
    for i in range(n):
        a += methodMC2(n-1)
    return a
```
Which of the following statements about the run time \\(T(n)\\) of `methodMC2` is **true**?
[] \\(T(n)\\) is \\(O(n)\\).
[] \\(T(n)\\) is \\(\Theta(n^3)\\).
[] \\(T(n) = c + dn + en^2\\) for some constants \\(c,d,e\\).
[x] \\(T(n) = c + dnT(n-1)\\) with \\(T(0) = e\\) for some constants \\(c,d,e\\).

Question 5:
Which of the following about the used space \\(S(n)\\) of `methodMC2` from the previous question is **true**?
[] \\(S(n) = T(n)\\).
[] \\(S(n)\\) is \\(\Theta(1)\\).
[x] \\(S(n)\\) is \\(\Theta(n)\\).
[] \\(S(n)\\) is \\(\Theta(n!)\\).

Solution 5:
\\(S(n) = c + dS(n-1)\\). Unfolding this we get: \\(S(n)\\) is \\(O(n)\\). Remember that we can reuse the space, so it is not
\\(nS(n-1)\\)!
