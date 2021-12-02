You should write a function `down_heap` that restores the heap property in a max-heap that is represented as a `List[int]`.

A heap is represented as a `List` in the following way.
The first element in the list represents the root of the heap.
The two following elements represent the two children of the root.
The following four elements represent the "grandchildren" of the root, etcetera.
Since a heap is by definition a complete binary tree, there will never be gaps in the list.
See the visible tests for some small examples.

Your function receives two arguments:

- `heap`, which is the list that represents the heap that you should restore the heap property for.
- `root`, which is the root at which the tree might be invalid. This might be a subtree of the overall heap.
  When the heap property is invalid at the root,
  the function fixes the heap first locally before fixing the affected subtree.
