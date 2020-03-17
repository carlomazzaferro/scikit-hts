from typing import List, Optional, Union

import pandas
from pmdarima import AutoARIMA
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.special._ufuncs import inv_boxcox
from scipy.stats import boxcox

from hts import HierarchyTree
from hts._t import Model, Transform
from hts.exceptions import InvalidArgumentException
from hts.transforms import FunctionTransformer


class TimeSeriesModel(BaseEstimator, RegressorMixin):
    """ Base class for the implementation of the underlying models.
        Inherits from scikit-learn base classes
    """

    def __init__(self,
                 kind: str,
                 node: HierarchyTree,
                 transform: Optional[Union[Transform, bool]] = None,
                 **kwargs):
        """
        Parameters
        ----------
        kind : str
            One of `prophet`, `sarimax`, `auto-arima`, `holt-winters`
        node : HierarchyTree
            Node
        transform : Bool or NamedTuple
        kwargs
            Keyword arguments to be passed to the model instantiation. See the documentation
            of each of the actual model implementations for a more comprehensive treatment
        """

        if kind not in Model.names():
            raise InvalidArgumentException(f'Model {kind} not valid. Pick one of: {" ".join(Model.names())}')

        self.kind = kind
        self.node = node
        self.model = self.create_model(**kwargs)
        self.forecast = None
        self.residual = None
        self.mse = None

        self.transform = transform
        if self.transform:
            if transform is True:
                self.transformer = FunctionTransformer(func=boxcox,
                                                       inv_func=inv_boxcox)
            else:
                self.transformer = FunctionTransformer(func=transform.func,
                                                       inv_func=transform.inv_func)
        else:
            self.transformer = FunctionTransformer(func=lambda x: (x, None),
                                                   inv_func=lambda x: (x, None))

    def create_model(self, **kwargs):

        if self.kind == Model.holt_winters.name:
            data = self.node.item
            model = ExponentialSmoothing(endog=data, **kwargs)

        elif self.kind == Model.auto_arima.name:
            model = AutoARIMA(**kwargs)

        elif self.kind == Model.sarimax.name:
            as_df = self._reformat(self.node.item)
            end = as_df[self.node.key]
            if self.node.exogenous:
                ex = as_df[self.node.exogenous]
            else:
                ex = None
            model = SARIMAX(endog=end, exog=ex, **kwargs)
        else:
            raise
        return model

    def _reformat(self, df):
        if isinstance(self.node.item, pandas.Series):
            df = pandas.DataFrame(self.node.item)
        else:
            df = self.node.item
        return df

    def fit(self, **fit_args) -> 'TimeSeriesModel':
        raise NotImplementedError

    def predict(self, node: HierarchyTree, **predict_args):
        raise NotImplementedError

    def fit_predict(self, node: HierarchyTree, **kwargs):
        return self.fit().predict(node)
