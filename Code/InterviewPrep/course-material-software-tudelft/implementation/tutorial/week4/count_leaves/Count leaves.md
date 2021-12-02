In this exercise you have to implement the function `count_leaves`, which returns the amount of leaf nodes in a binary tree.
Leaf nodes are nodes that have no child nodes.

For example, for the following tree the answer is 2 (as nodes 2 and 3 are leaves):
```
    1
  /   \
 2     3
```

For the following tree, the answer is 3 (as nodes 2, 5 and 6 are leaves):
```
    1
  /   \
2      3
     /   \
   4      5
 /
6 
```

These examples are available in the tests.

In the library you will find an implementation for the `BinaryTree` class.
Note that the left and right child of a binary tree are trees themselves.
