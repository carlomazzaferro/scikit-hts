import logging

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

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, X):
        try:
            for col in X.columns[1:]:
                X[col], self.lam = self.func(X[col])
        except RuntimeWarning as e:
            logger.error(f'Coulnd\'t perform transform: {repr(e)}')
        return X

    def fit_transform(self, X, y=None, **fit_params):
        try:
            for col in X.columns[1:]:
                X[col], self.lam = self.func(X[col], **fit_params)
        except RuntimeWarning as e:
            logger.error(f'Coulnd\'t perform transform: {repr(e)}')
        return X

    def inverse_transform(self, X):
        if isinstance(X, pandas.Series):
            return self.inv_func(X, self.lam)
        try:
            for col in X.columns[1:]:
                X[col] = self.inv_func(X[col], self.lam)
        except RuntimeWarning as e:
            logger.error(f'Coulnd\'t perform transform: {repr(e)}')
        return X




