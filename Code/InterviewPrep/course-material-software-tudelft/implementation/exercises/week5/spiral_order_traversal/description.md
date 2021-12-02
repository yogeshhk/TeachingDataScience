In this exercise you will implement the spiral order traversal. The function `traversal` will take as input a binary tree and return a list with the values of the nodes in the correct order.

Spiral order traversal works as follows, in the odd levels of the tree, you will iterate from left to right, in the even levels of the tree, you will iterate from right to left.

For example given the following tree:

```
        1           level 1
      /   \     
     /     \
    2       3       level 2
   / \     / \
  4   5   6   7     level 3
```

The resulting list will be `[1, 3, 2, 4, 5, 6, 7]`

A more visual example:

![Spiral order tree traversal](https://tekmarathon.files.wordpress.com/2013/06/spriralbst2.jpg)

(source: [https://tekmarathon.com/2013/06/11/spiral-traversal-of-binary-tree/](https://tekmarathon.com/2013/06/11/spiral-traversal-of-binary-tree/))