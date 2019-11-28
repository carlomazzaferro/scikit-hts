
import weakref


class NTree(object):
    """
    A generic N-ary tree implementations, that uses a list to store
    it's children.
    """

    def __init__(self, key=None, item=None, children=None, parent=None):
        self.key = key
        self.item = item
        self.children = children or []
        self._parent = weakref.ref(parent) if parent else None

    @property
    def parent(self):
        if self._parent:
            return self._parent()

    def __getstate__(self):
        self._parent = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        for child in self.children:
            child._parent = weakref.ref(self)

    def __str__(self):
        return '{} : {}'.format(self.key, self.item)

    def is_leaf(self):
        return len(self.children) == 0

    def get_height(self):
        heights = [child.get_height() for child in self.children]
        return max(heights) + 1 if heights else 1

    def traversal(self, visit=None, *args, **kwargs):
        visit(self, *args, **kwargs)
        l = [self]
        for child in self.children:
            l += child.traversal(visit, *args, **kwargs)
        return l

    def __iter__(self):
        yield self
        for child in self.children:
            yield child

    def add_child(self, key=None, item=None):
        child = NTree(key=key, item=item, parent=self)
        self.children.append(child)
        return child
