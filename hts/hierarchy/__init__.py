from __future__ import annotations

import weakref
from collections import deque
from itertools import chain
from typing import Tuple, Union, List, Optional, Dict

import pandas

from hts._t import NAryTreeT
from hts.hierarchy.utils import fetch_cols, hexify, groupify, resample_count
from hts.viz.geo import HierarchyVisualizer


class HierarchyTree(NAryTreeT):
    """
    A generic N-ary tree implementations, that uses a list to store
    it's children.
    """

    @classmethod
    def from_geo_events(cls,
                        df: pandas.DataFrame,
                        lat_col: str,
                        lon_col: str,
                        nodes: Tuple,
                        levels: Tuple[int, int] = (6, 7),
                        resample_freq: str = '1H',
                        min_count: Union[float, int] = 0.2,
                        root_name: str = 'total'
                        ):
        """

        Parameters
        ----------
        df
        lat_col
        lon_col
        nodes
        levels
        resample_freq
        min_count
        root_name

        Returns
        -------

        """

        hexified = hexify(df, lat_col, lon_col, levels=levels)
        total = resample_count(hexified, resample_freq, root_name)
        hierarchy = cls(key=root_name, item=total)
        return groupify(hierarchy,
                        df=hexified,
                        nodes=nodes,
                        freq=resample_freq,
                        min_count=min_count,
                        total=total)

    @classmethod
    def from_nodes(cls,
                   nodes: Dict[str, List[str]],
                   df: pandas.DataFrame,
                   exogenous: Dict[str, List[str]] = None,
                   root: Union[str, HierarchyTree] = 'total',
                   top: HierarchyTree = None,
                   stack: List = None):
        """
        Standard method for creating a hierarchy from nodes and a dataframe containing as columns those nodes.
        The nodes are represented as a dictionary containing as keys the nodes, and as values list of edges.
        See the examples for usage.

        Parameters
        ----------
        nodes
        df
        exogenous
        root
        top
        stack

        Returns
        -------
        hierarchy : HierarchyTree
            The hierarchy tree representation of your data

        Examples
        --------

        >>> from hts.utils import load_hierarchical_sine_data
        >>> from datetime import datetime
        >>> from hts import HierarchyTree

        >>> s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
        >>> dti = load_hierarchical_sine_data(s, e)  # Create some sample data
        >>> hier = {'total': ['a', 'b', 'c'], 'a': ['aa', 'ab'], 'aa': ['aaa', 'aab'], 'b': ['ba', 'bb'], 'c': ['ca', 'cb', 'cc', 'cd']}
        >>> ht = HierarchyTree.from_nodes(hier, dti)
        >>> print(ht)
        - total
           |- a
           |  |- aa
           |  |  |- aaa
           |  |  - aab
           |  - ab
           |- b
           |  |- ba
           |  - bb
           - c
              |- ca
              |- cb
              |- cc
              - cd
        """

        if stack is None:
            stack = []
            cols, ex = fetch_cols(exogenous, root)
            root = HierarchyTree(key=root, item=df[cols], exogenous=ex)  # root node is created
            top = root  # it is stored in b variable
        x = root.key  # root.val = 2 for the first time
        if len(nodes[x]) > 0:  # check if there are children of the node exists or not
            for i in range(len(nodes[x])):  # iterate through each child
                key = nodes[x][i]
                cols, ex = fetch_cols(exogenous, key)
                y = HierarchyTree(key=key, item=df[cols], exogenous=ex)  # create Node for every child
                root.children.append(y)  # append the child_node to its parent_node
                stack.append(y)  # store that child_node in stack
                if y.key not in nodes.keys():  # if the child_node_val = 0 that is the parent = leaf
                    stack.pop()  # pop the 0 value from the stack
            if len(stack):
                if len(stack) >= 2:   # bottom-to-top
                    tmp_root = stack.pop(0)
                else:
                    tmp_root = stack.pop()
                cls.from_nodes(nodes=nodes,
                               df=df,
                               exogenous=exogenous,
                               root=tmp_root,
                               top=top,
                               stack=stack)  # pass tmp_root to the function as root
        return top

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


if __name__ == '__main__':
    from hts.utils import hierarchical_sine_data
    from datetime import datetime

    s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
    dti = hierarchical_sine_data(s, e)
    hier = {'total': ['a', 'b', 'c'], 'a': ['aa', 'ab'], 'aa': ['aaa', 'aab'], 'b': ['ba', 'bb'], 'c': ['ca', 'cb', 'cc', 'cd']}

    ht = HierarchyTree.from_nodes(hier, dti)
    print(ht)