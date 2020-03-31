import logging
from typing import Union

import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.special import inv_boxcox
from scipy.stats import boxcox


logger = logging.getLogger(__name__)


class FunctionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,
                 func: callable = boxcox,
                 inv_func: callable = inv_boxcox):

        self.func = func
        self.inv_func = inv_func
        self.lam = None

    def fit(self, x: pandas.Series, y=None, **fit_params):
        return self

    def transform(self, x: pandas.Series):
        if any(x.values == 0):
            x += 1
            x, self.lam = self.func(x.values)
            return x
        elif any(x.values < 0):
            raise ValueError('Boxcox can\t be applied, column has negative values')
        else:
            x, self.lam = self.func(x.values)
            return x

    def fit_transform(self, x: pandas.Series, y=None, **fit_params):
        return self.fit(x).transform(x)

    def inverse_transform(self, x: Union[pandas.Series, numpy.ndarray]):
        if isinstance(x, pandas.Series):
            return self.inv_func(x.values, self.lam)
        else:
            return self.inv_func(x, self.lam)



