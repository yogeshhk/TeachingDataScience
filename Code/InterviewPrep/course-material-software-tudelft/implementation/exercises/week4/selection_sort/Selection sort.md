In this exercise you will implement in-place selection sort. The algorithm works as follows:

1. `x` is initialized with `0`.
2. Find the index of the element with the lowest value in the _implicit_ list slice `xs[x:]`. 
3. Swap the elements at location `x` with the minimum element found in step 2.
4. Repeat 2 and 3 with `x = x + 1` until the list is sorted.
