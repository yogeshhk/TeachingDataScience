In this exercise you will have to implement the function `search`, this function takes as input a multi way search tree, `MWSTree`, and a number. The result will be `True` if the number is stored in this multi way tree, `False` if it can't be found.

We have provided a simple multi way search tree. Every node in this tree has the following attributes:
- A list of keys, `keys`, these are the values stored in the node. 
- A list of children, `children`, these are the nodes that are the children of this node. Note that (in the case that a node is not a leaf) the list of children will always have one more element than the list of keys.

The tree looks like this:
```
         [10, 20, 30]
         /   |   |  \   
        /    |   |   \
       /     |   |    \
      /      |   |     \
     /       |   |      \
    /        |   |       \
[1, 2] [12, 15] [21, 26] [34]
```

The root contains the keys `[10, 20, 30]`, therefore it has 4 children.

**IMPORTANT**: the order in the lists are important, the leftmost child will be the first element in the list and leftmost key will always be the smallest key.