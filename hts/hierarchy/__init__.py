import weakref
from collections import deque
from itertools import chain
from typing import List, Optional, Tuple, Union

import pandas

from hts._t import ExogT, NAryTreeT, NodesT
from hts.hierarchy.utils import (
    fetch_cols,
    groupify,
    hexify,
    make_iterable,
    resample_count,
)
from hts.viz.geo import HierarchyVisualizer


class HierarchyTree(NAryTreeT):
    """
    A generic N-ary tree implementations, that uses a list to store
    it's children.
    """

    @classmethod
    def from_geo_events(
        cls,
        df: pandas.DataFrame,
        lat_col: str,
        lon_col: str,
        nodes: Tuple,
        levels: Tuple[int, int] = (6, 7),
        resample_freq: str = "1H",
        min_count: Union[float, int] = 0.2,
        root_name: str = "total",
        fillna: bool = False,
    ):
        """

        Parameters
        ----------
        df : pandas.DataFrame
        lat_col : str
            Column where the latitude coordinates can be found
        lon_col : str
            Column where the longitude coordinates can be found
        nodes : str

        levels :
        resample_freq
        min_count
        root_name
        fillna

        Returns
        -------
        HierarchyTree
        """

        hexified = hexify(df, lat_col, lon_col, levels=levels)
        total = resample_count(hexified, resample_freq, root_name)
        hierarchy = cls(key=root_name, item=total)
        grouped = groupify(
            hierarchy,
            df=hexified,
            nodes=nodes,
            freq=resample_freq,
            min_count=min_count,
            total=total,
        )
        # TODO: more flexible strategy
        if fillna:
            df = grouped.to_pandas()
            df = df.fillna(method="ffill").dropna()
            for node in make_iterable(grouped, prop=None):
                repl = df[[node.key]]
                node.item = repl
        return grouped

    @classmethod
    def from_nodes(
        cls,
        nodes: NodesT,
        df: pandas.DataFrame,
        exogenous: ExogT = None,
        root: Union[str, "HierarchyTree"] = "total",
        top: "HierarchyTree" = None,
        stack: List = None,
    ):
        """
        Standard method for creating a hierarchy from nodes and a dataframe containing as columns those nodes.
        The nodes are represented as a dictionary containing as keys the nodes, and as values list of edges.
        See the examples for usage. The total column must be named total and not something else.

        Parameters
        ----------
        nodes : NodesT
            Nodes definition. See ``Examples``.
        df : pandas.DataFrame
            The actual data containing the nodes
        exogenous : ExogT
            The nodes representing the exogenous variables
        root : Union[str, HierarchyTree]
            The name of the root node
        top : HierarchyTree
            Not to be used for initialisation, only in recursive calls
        stack : list
            Not to be used for initialisation, only in recursive calls

        Returns
        -------
        hierarchy : HierarchyTree
            The hierarchy tree representation of your data

        Examples
        --------
        In this example we will create a tree from some multivariate data

        >>> from hts.utilities.load_data import load_mobility_data
        >>> from hts.hierarchy import HierarchyTree

        >>> hmv = load_mobility_data()
        >>> hmv.head()
                    WF-01  CH-07  BT-01  CBD-13  SLU-15  CH-02  CH-08  SLU-01  BT-03  CH-05  SLU-19  SLU-07  SLU-02  CH-01  total   CH  SLU  BT  OTHER  temp  precipitation
        starttime
        2014-10-13     16     14     20      16      20     42     24      24     12     22      14       2       8      6    240  108   68  32     32  62.0           0.00
        2014-10-14     22     28     28      38      36     36     42      40     14     26      18      32      16     18    394  150  142  42     60  59.0           0.11
        2014-10-15     10     14      8      20      18     38     16      28     18     10       0      24      10     16    230   94   80  26     30  58.0           0.45
        2014-10-16     22     18     24      44      44     40     24      20     22     18       8      26      14     14    338  114  112  46     66  61.0           0.00
        2014-10-17      8     12     16      20      18     22     32      12      8     28      10      30       8     10    234  104   78  24     28  60.0           0.14

        >>> hier = {
                'total': ['CH', 'SLU', 'BT', 'OTHER'],
                'CH': ['CH-07', 'CH-02', 'CH-08', 'CH-05', 'CH-01'],
                'SLU': ['SLU-15', 'SLU-01', 'SLU-19', 'SLU-07', 'SLU-02'],
                'BT': ['BT-01', 'BT-03'],
                'OTHER': ['WF-01', 'CBD-13']
            }
        >>> exogenous = {k: ['precipitation', 'temp'] for k in hmv.columns if k not in ['precipitation', 'temp']}
        >>> ht = HierarchyTree.from_nodes(hier, hmv, exogenous=exogenous)
        >>> print(ht)
        - total
           |- CH
           |  |- CH-07
           |  |- CH-02
           |  |- CH-08
           |  |- CH-05
           |  - CH-01
           |- SLU
           |  |- SLU-15
           |  |- SLU-01
           |  |- SLU-19
           |  |- SLU-07
           |  - SLU-02
           |- BT
           |  |- BT-01
           |  - BT-03
           - OTHER
              |- WF-01
              - CBD-13

        """

        if stack is None:
            stack = []
            cols, ex = fetch_cols(exogenous, root)
            root = HierarchyTree(
                key=root, item=df[cols], exogenous=ex
            )  # root node is created
            top = root  # it is stored in b variable
        x = root.key  # root.val = 2 for the first time
        if len(nodes[x]) > 0:  # check if there are children of the node exists or not
            for i in range(len(nodes[x])):  # iterate through each child
                key = nodes[x][i]
                cols, ex = fetch_cols(exogenous, key)
                y = HierarchyTree(
                    key=key, item=df[cols], exogenous=ex
                )  # create Node for every child
                root.children.append(y)  # append the child_node to its parent_node
                stack.append(y)  # store that child_node in stack
                if (
                    y.key not in nodes.keys()
                ):  # if the child_node_val = 0 that is the parent = leaf
                    stack.pop()  # pop the 0 value from the stack
            if len(stack):
                if len(stack) >= 2:  # bottom-to-top
                    tmp_root = stack.pop(0)
                else:
                    tmp_root = stack.pop()
                cls.from_nodes(
                    nodes=nodes,
                    df=df,
                    exogenous=exogenous,
                    root=tmp_root,
                    top=top,
                    stack=stack,
                )  # pass tmp_root to the function as root
        return top

    def __init__(
        self,
        key: str = None,
        item: Union[pandas.Series, pandas.DataFrame] = None,
        exogenous: List[str] = None,
        children: List[NAryTreeT] = None,
        parent: NAryTreeT = None,
    ):

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
        """
        Get a node given its key

        Parameters
        ----------
        key: str
            The key of the node of interest
        Returns
        -------
        node : HierarchyTree
            The node of interest

        """
        for node in self.traversal_level():
            if node.key == key:
                return node
        return None

    def traversal_level(self) -> List[NAryTreeT]:
        """
        Level order traversal of the tree

        Returns
        -------
        list of nodes
        """

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
        """
        Return the of nodes in the tree

        Returns
        -------
        num nodes : int
        """

        return sum(chain.from_iterable(self.level_order_traversal()))

    def is_leaf(self):
        """
        Check if node is a leaf Node

        Returns
        -------
        bool
            True or False
        """

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
        """
        Iterate through the tree in level order, getting the number of children for
        each node

        Returns
        -------
        list[list[int]]
        """

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

    def get_level_order_labels(self: NAryTreeT) -> List[List[str]]:
        """
        Get the associated node labels from the NAryTreeT level_order_traversal().

        Parameters
        ----------
        self: NAryTreeT
            Tree being searched.

        Returns
        -------
        List[List[str]]
            Node labels corresponding to level order traversal.
        """
        labels = []
        q = deque([(self, 0)])
        while q:
            n, li = q.popleft()
            if len(labels) < li + 1:
                labels.append([])
            for i in n.children:
                q.append((i, li + 1))
            labels[li].append(n.key)
        return labels

    def add_child(self, key=None, item=None, exogenous=None) -> NAryTreeT:
        child = HierarchyTree(key=key, item=item, exogenous=exogenous, parent=self)
        self.children.append(child)
        return child

    def leaf_sum(self) -> int:
        return sum(self.level_order_traversal()[-1])

    def to_pandas(self) -> pandas.DataFrame:
        """
        Transforms the hierarchy into a pandas.DataFrame
        Returns
        -------
        df : pandas.DataFrame
            Dataframe representation of the tree
        """
        df = pandas.concat(
            [self.item] + [c.item[c.key] for c in self.traversal_level()], 1
        )
        df.index.name = "ds"
        return df

    def get_series(self) -> pandas.Series:
        return self.item[self.key]
