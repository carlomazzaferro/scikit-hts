import logging
from typing import Union

import numpy
import pandas
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lam = None

    def fit(self, x: pandas.Series, y=None, **fit_params):
        return self

    def transform(self, x: pandas.Series):
        if any(x.values == 0):
            x += 1
            x, self.lam = boxcox(x.values)
            return x
        elif any(x.values < 0):
            raise ValueError("Boxcox can\t be applied, column has negative values")
        else:
            x, self.lam = boxcox(x.values)
            return x

    def fit_transform(self, x: pandas.Series, y=None, **fit_params):
        return self.fit(x).transform(x)

    def inverse_transform(self, x: Union[pandas.Series, numpy.ndarray]):
        if isinstance(x, pandas.Series):
            return inv_boxcox(x.values, self.lam)
        else:
            return inv_boxcox(x, self.lam)


class FunctionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, func: callable = None, inv_func: callable = None):
        if not func or not inv_func:
            raise ValueError("`func` and `inv_func` must be passed")
        self.func = func
        self.inv_func = inv_func

    def fit(self, x: pandas.Series, y=None, **fit_params):
        return self

    def transform(self, x: pandas.Series):
        return self.func(x.values)

    def fit_transform(self, x: pandas.Series, y=None, **fit_params):
        return self.fit(x).transform(x)

    def inverse_transform(self, x: Union[pandas.Series, numpy.ndarray]):
        return self.inv_func(x)
