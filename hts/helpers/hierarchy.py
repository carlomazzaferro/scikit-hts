from __future__ import annotations
import weakref
from collections import deque
from typing import List


import pandas

from hts.types import NAryTreeT
from hts.viz.geo import HierarchyVisualizer
from hts.helpers import flatten


class HierarchyTree(NAryTreeT):
    """
    A generic N-ary tree implementations, that uses a list to store
    it's children.
    """

    def __init__(self, key=None, item=None, children=None, parent=None):
        self.key = key
        self.item = item
        self.children = children or []
        self._parent = weakref.ref(parent) if parent else None
        self.visualizer = HierarchyVisualizer(self)

    def traversal(self) -> List[NAryTreeT]:
        l = [self]
        for child in self.children:
            l += child.traversal()
        return l

    def num_nodes(self) -> int:
        return sum(flatten(self.level_order_traversal()))

    def is_leaf(self):
        return len(self.children) == 0

    def value_at_height(self, level: int) -> List:
        if level == 0:
            return [1]
        return self.level_order_traversal()[level - 1]

    def sum_at_height(self, level) -> int:
        return sum(self.value_at_height(level))

    def get_height(self) -> int:
        return len(self.level_order_traversal())

    def level_order_traversal(self: NAryTreeT) -> List[List[int]]:
        if self is None:
            return []
        res = []
        q = deque([(self, 0)])
        while q:
            n, li = q.popleft()
            if len(res) < li + 1:
                res.append([])
            for i in n.children:
                q.append((i, li + 1))
            res[li].append(len(n.children))
        return res[:-1]

    def add_child(self, key=None, item=None) -> NAryTreeT:
        child = HierarchyTree(key=key, item=item, parent=self)
        self.children.append(child)
        return child

    def leaf_sum(self) -> int:
        return sum(self.level_order_traversal()[-1])

    def to_pandas(self):
        df = pandas.concat([c.item for c in self.traversal()], 1)
        df.index.name = 'ds'
        # df = df.reset_index()
        return df
