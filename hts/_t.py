import abc
import logging
import weakref
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    NewType,
    Optional,
    Tuple,
    Union,
)
from weakref import ReferenceType

import numpy
import pandas
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class ExtendedEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))


class ModelT(str, ExtendedEnum):
    prophet = "prophet"
    holt_winters = "holt_winters"
    auto_arima = "auto_arima"
    sarimax = "sarimax"


class UnivariateModelT(str, ExtendedEnum):
    arima = "arima"
    auto_arima = "auto_arima"
    prophet = "prophet"
    holt_winters = "holt_winters"
    sarimax = "sarimax"


Models = NewType("ModelT", ModelT)


class Transform(NamedTuple):
    func: Callable
    inv_func: Callable


class HierarchyVisualizerT(metaclass=abc.ABCMeta):
    tree: "NAryTreeT"

    def create_map(self):
        ...


class NAryTreeT(metaclass=abc.ABCMeta):
    """
    Type definition of an NAryTree
    """

    key: str
    item: Union[pandas.Series, pandas.DataFrame]
    exogenous: List[str] = None
    children: List[Optional["NAryTreeT"]]
    _parent: "Optional[ReferenceType[NAryTreeT]]"
    visualizer: HierarchyVisualizerT

    @property
    def parent(self):
        if self._parent:
            return self._parent()

    def __iter__(self) -> "NAryTreeT":
        yield self
        for child in self.children:
            yield child

    def __getstate__(self):
        self._parent = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        for child in self.children:
            child._parent = weakref.ref(self)

    def num_nodes(self) -> int:
        ...

    def is_leaf(self) -> bool:
        ...

    def value_at_height(self, level: int) -> List:
        ...

    def sum_at_height(self, level) -> int:
        ...

    def get_height(self) -> int:
        ...

    def level_order_traversal(self: "NAryTreeT") -> List[List[int]]:
        ...

    def get_level_order_labels(self: "NAryTreeT") -> List[List[str]]:
        ...

    def traversal_level(self) -> List["NAryTreeT"]:
        ...

    def add_child(self, key=None, item=None, exogenous=None) -> "NAryTreeT":
        ...

    def leaf_sum(self) -> int:
        ...

    def to_pandas(self) -> pandas.DataFrame:
        ...

    def get_node_height(self, key: str) -> int:
        ...

    def get_series(self) -> pandas.Series:
        ...

    def string_repr(self, prefix="", _last=True):
        base = "".join([prefix, "- " if _last else "|- ", self.key, "\n"])
        prefix += "   " if _last else "|  "
        child_count = len(self.children)
        for i, child in enumerate(self.children):
            _last = i == (child_count - 1)
            base += child.string_repr(_last=_last, prefix=prefix)
        return base

    def __repr__(self):
        return self.string_repr()

    __str__ = __repr__


class TimeSeriesModelT(BaseEstimator, RegressorMixin, metaclass=abc.ABCMeta):

    """ Type definition of an TimeSeriesModel """

    kind: str
    node: NAryTreeT
    model: Any
    forecast: Optional[pandas.DataFrame]
    residual: Optional[numpy.ndarray]
    mse: Optional[numpy.ndarray]
    transform: "TransformT"

    @staticmethod
    def _no_func(x):
        return x

    def _set_results_return_self(self, in_sample, y_hat) -> "TimeSeriesModelT":
        ...

    def create_model(self, **kwargs):
        ...

    def fit(self, **fit_args) -> "TimeSeriesModelT":
        raise NotImplementedError

    def predict(self, node: NAryTreeT, **predict_args):
        raise NotImplementedError


class MethodT(ExtendedEnum):
    OLS = "OLS"
    WLSS = "WLSS"
    WLSV = "WLSV"
    FP = "FP"
    PHA = "PHA"
    AHP = "AHP"
    BU = "BU"
    NONE = "NONE"


# TODO: make this a proper recursive type when mypy supports it: https://github.com/python/mypy/issues/731
HierarchyT = Tuple[str, "HierarchyT"]
NodesT = ExogT = Dict[str, List[str]]
LowMemoryFitResultT = Tuple[str, str]
ModelFitResultT = Union[TimeSeriesModelT, LowMemoryFitResultT]
HTSFitResultT = List[ModelFitResultT]
TransformT = Union[Transform, bool]
ArrayLike = Union[numpy.ndarray, pandas.Series, pandas.DataFrame]
