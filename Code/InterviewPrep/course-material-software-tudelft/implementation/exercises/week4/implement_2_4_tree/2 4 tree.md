In this exercise you will implement some functions of a 2-4 tree. A `Tree` and `Node` class are provided together with the implementations of some functions.

Your task is to implement `Node.split`, `Tree.contains` and `Tree.add`.

- `Node.split` splits the current node. This function will be used in `Tree.add`. Remember that we only split 4-nodes.
- `Tree.add` adds the given value to the tree in \\(\mathcal{O} log n\\) time where \\(n\\) is the amount of nodes in the tree.
- `Tree.contains` returns `True` if the given value is in the tree, `False` otherwise. It must operate in \\(\mathcal{O} log n\\) time where \\(n\\) is the amount of nodes in the tree.

<details>
    <summary>Insertion into a 2-4 tree</summary>
    The insertion operation of a 2-4 tree is visualised on <a href="https://www.educative.io/page/5689413791121408/80001">this page</a>. We start from the root and find the leaf node that should contain the new value. Along the way we split all 4-nodes. Then we add the value to the found leaf node.
</details>
