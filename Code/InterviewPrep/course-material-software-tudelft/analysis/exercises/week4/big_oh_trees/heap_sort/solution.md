1) Derive the run time equation of this code and explain all the terms.

We take \\(n\\) to be the length of `xs`.

This is an implementation of heap sort. We know that heaps are balanced, therefore adding and removing an item can be done in \\(\mathcal{O}(log n)\\) time.

\\(T(n)=c_0 + n c_1 + 2n \cdot \mathcal{O}(log n)\\) where:

- \\(c_0\\) represents initialising the heap (line 2)
- \\(c_1\\) represents getting the next item in the 2 loops (lines 3 and 5) and initialising the result list (line 5)
- \\(2n \cdot \mathcal{O}(log n)\\) represents \\(n\\) `add` operations on the heap and \\(n\\) `remove_min` operations, which both take \\(\mathcal{O}(log n)\\) time.
  
2) State the run time complexity of the function in terms of Big Oh notation. Explain your answer, but a full proof is not needed.​

\\(T(n)=c_0 + n c_1 + 2n \cdot \mathcal{O}(log n)\\)

After simplification, the constant terms and factors are removed. This leaves us with \\(T(n) = n + n \cdot \mathcal{O}(log n)\\), which is linearithmic growth. Thus, the run time complexity is \\(\mathcal{O}(n log n)\\).