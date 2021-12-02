In the lectures you have learned that the fastest we can sort using a comparison based sorting algorithm is \\(\mathcal{O}(n log_2 n)\\).
In this exercise you will implement a sorting algorithm that can run faster, depending on the input size.

Count sort is not comparison based and only works for lists of integer values. It runs in \\(\mathcal{O}(n+m)\\), where \\(m\\) is
the length of the range of all values (i.e. the largest value - the lowest value + 1).
Depending on the size of the range, this means count sort can be a lot faster than for example merge sort.

The algorithm works as follows:

1. Determine the range of the values in the input list.
2. Create a list of `0`s with the same length as the range found in 1 (we call it `count`).
3. For every element in the input list, increment the according value in `count` by 1.  
    - A value `x` in the input list is mapped to `count[x - min_value]`.
4. Create an empty result list `res`.
5. For every element `c` in `count`, add the value it represents to `res` `c` times.
6. Return `res`.

Implement the algorithm according to these steps. Make sure your code has a complexity of \\(\mathcal{O}(n+m)\\).

**Note**: for the fastest performance, you can use the following optimisations:

- `some_list += another_list` is faster than `some_list = some_list + another_list`
- To add the same item to a list `x` times, the fastest way is `some_list += [item] * x`

<details>
    <summary>Why is that faster? (extra information)</summary>
    We do not assume this knowledge, but if you are interested: this section explains why the optimisations mentioned above are faster.<br/>
    <br/>
    <h5><code>a += b</code> instead of <code>a = a + b</code></h5>
    The <code>a += b</code> operator for lists works by extending the list <code>a</code> with a list <code>b</code>. This grows <code>a</code> by <code>len(b)</code> and then adds all elements.<br/>
    With <code>a = a + b</code> we first create a new list of size <code>len(a) + len(b)</code>, then copy all elements from <code>a</code> and <code>b</code> in there and reassign it to <code>a</code>. As you can see this means we are doing <code>len(a)</code> more additions to a list than with <code>a += b</code>.
    Note that <code>a += b</code> is (largely) the same as <code>a.extend(b)</code>.<br/>
    <br/>
    <h5><code>a += [item] * x</code> instead of appending in a loop</h5>
    Under the hood python optimizes extending of a list: extra space is allocated to accomodate for the new part of the list. This means the items in the list <code>[item] * x</code> can be added without having to grow <code>a</code> multiple times.<br/>
    Furthermore the loop used in <code>extend</code> is implemented in the C language (in which python is implemented), which is significantly faster than the loop we would use in native python around <code>append</code>.
</details>