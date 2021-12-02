Implement the function `even_before_odd` that recursively reorders the values in a list of integer values so that all even values end up before all odd values. The order of the numbers does not matter.

You need to modify the original list to be in the correct order, not return a new list.

Hint: it may be useful to add a helper function.

If you get stuck, the hint below describes a possible solution.

<details>
    <summary>An algorithm that solves this problem</summary>
    This algorithm keeps track of the list and 2 pointers: the lower and upper bounds.
    The part of the list between these pointers is the part that needs to be ordered (that is: all elements in <code>xs[lower:upper]</code>, i.e. including the lower bound, excluding the upper bound). Let's look at an example:
<pre>
| 1 | 2 | 3 | 4 |
  ^               ^</pre>
    The algorithm looks at the value at the lower pointer. This is 1, so odd. We move it to the end of the list.
    We do this by removing it from the beginning of the list and adding it at the back. Then we move the upper bound pointer,
    as we know this element is now in the correct location.
<pre>
| 2 | 3 | 4 | 1 |
  ^           ^</pre>
    Currently the value at the lower pointer is 2, so even. We leave this and move the lower bound pointer.
<pre>
| 2 | 3 | 4 | 1 |
      ^       ^</pre>
    The 3 is again moved to the back, after which the upper bound pointer is moved.
<pre>
| 2 | 4 | 1 | 3 |
      ^   ^</pre>   
    Last step: 4 is even, so we move the lower bound pointer.
<pre>
| 2 | 4 | 1 | 3 |
          ^
          ^</pre>   
    There are no more items to check, the algorithm terminates with a correct solution.

</details>