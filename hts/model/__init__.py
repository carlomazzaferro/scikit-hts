from typing import List, Optional, Union

import pandas
from pmdarima import AutoARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.special._ufuncs import inv_boxcox
from scipy.stats import boxcox

from hts import HierarchyTree
from hts._t import Model, Transform
from hts.exceptions import InvalidArgumentException
from hts.transforms import FunctionTransformer


class TimeSeriesModel:
    """
    Parameters
    ----------
    kind : str
        One of 'prophet', 'arima', 'auto-arima', 'varmax', 'holt-winters'
    data : pandas.DataFrame
        The data to which to fit the model
    endogenous : List[str]
        list of columns specifying the endogenous variables
    exogenous : Union[str, List[str]]
        Columns specifying the exogenous variables
    kwargs :
    """

    def __init__(self,
                 kind: str,
                 node: HierarchyTree,
                 transform: Optional[Union[Transform, bool]] = None,
                 **kwargs):

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

        elif self.kind == Model.arima.name:
            return AutoARIMA(**kwargs)

        elif self.kind == Model.sarimax.name:
            data = self.node.item[self.node.key]
            ex = self.node.item[self.node.exogenous]
            return SARIMAX(endog=data, exog=ex, **kwargs)
        else:
            raise
        return model

    def fit_predict(self, **kwargs):
        raise NotImplementedError
