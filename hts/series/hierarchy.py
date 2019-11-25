from __future__ import annotations

import attr
import pandas


class Node(object):
    def __init__(self, children=None):
        self.children = children


class RootNone(Node):
    def __init__(self):
        super().__init__()


class HierarchicalSeries(object):
    def __init__(self):
        ...

