from __future__ import annotations
from typing import List


class MWSTreeNode:

    def __init__(self, keys: List[int], children: List[MWSTreeNode]):
        self.keys = keys
        self.children = children


class MWSTree:
    def __init__(self):
        self.root = MWSTreeNode([], [])
