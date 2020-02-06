from __future__ import annotations

import weakref
from collections import deque
from itertools import chain
from typing import Tuple, Union, List, Optional

import pandas

from hts._t import NAryTreeT
from hts.viz.geo import HierarchyVisualizer


class HierarchyTree(NAryTreeT):
    """
    A generic N-ary tree implementations, that uses a list to store
    it's children.
    """

    @classmethod
    def from_events(cls,
                    df: pandas.DataFrame,
                    lat_col: str,
                    lon_col: str,
                    nodes: Tuple,
                    levels: Tuple[int, int] = (6, 7),
                    resample_freq: str = '1H',
                    min_count: Union[float, int] = 0.2,
                    ):

        from hts.helpers import hexify, groupify
        hexified = hexify(df, lat_col, lon_col, levels=levels)
        return groupify(hexified, nodes=nodes, freq=resample_freq, min_count=min_count)

    @classmethod
    def from_df(cls,
                df: pandas.DataFrame,
                nodes: Tuple,
                ):
        pass

    def __init__(self,
                 key: str = None,
                 item: Union[pandas.Series, pandas.DataFrame] = None,
                 exogenous: List[str] = None,
                 children: List[NAryTreeT] = None,
                 parent: NAryTreeT = None):

        self.key = key
        self.item = item
        if exogenous:
            self.exogenous = exogenous
        else:
            self.exogenous = []
        self.children = children or []
        self._parent = weakref.ref(parent) if parent else None
        self.visualizer = HierarchyVisualizer(self)

    def get_node(self, key: str) -> Optional[NAryTreeT]:
        for node in self.traversal_level():
            if node.key == key:
                return node
        return None

    def traversal_level(self) -> List[NAryTreeT]:
        res = []
        q = deque([(self, 0)])
        while q:
            n, li = q.popleft()
            if len(res) < li + 1:
                res.append([])
            for i in n.children:
                q.append((i, li + 1))
            res[li].extend(n.children)
        return list(chain.from_iterable(res[:-1]))

    def num_nodes(self) -> int:
        return sum(chain.from_iterable(self.level_order_traversal()))

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

    def get_node_height(self, key: str) -> int:
        for node in self.traversal_level():
            if node.key == key:
                return node.get_height()
        return -1

    def level_order_traversal(self: NAryTreeT) -> List[List[int]]:
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

    def add_child(self, key=None, item=None, exogenous=None) -> NAryTreeT:
        child = HierarchyTree(key=key, item=item, exogenous=exogenous, parent=self)
        self.children.append(child)
        return child

    def leaf_sum(self) -> int:
        return sum(self.level_order_traversal()[-1])

    def to_pandas(self):
        df = pandas.concat([self.item] + [c.item for c in self.traversal_level()], 1)
        df.index.name = 'ds'
        return df
