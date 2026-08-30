class Tree:

    def getleaves(self) -> List[TreeNode]:
        return self.helper(self.root)

    def helper(self, node: TreeNode) -> List[TreeNode]:
        if len(node.children) == 0:
            return [node]  # This is a leaf!
        else:
            res = []
            for c in node.children:
                res.extend(self.helper(c))
            return res
