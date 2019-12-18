from typing import Sequence, List, Union, Optional

import pandas
from fbprophet import Prophet
from pmdarima import AutoARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from hts.core.types import Model, UnivariateModel, MultivariateModel
from hts.exceptions import InvalidArgumentException


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
                 data: pandas.DataFrame,
                 endogenous: Union[str, List[str]] = 'y',
                 exogenous: Optional[List[str]] = None,
                 **kwargs):

        if kind not in Model.names():
            raise InvalidArgumentException(f'Model {kind} not valid. Pick one of: {" ".join(Model.names())}')

        if isinstance(endogenous, list):
            if kind in UnivariateModel.names():
                raise InvalidArgumentException(f'Multiple endogenous variables are allowed only for multivariate models:'
                                               f'{" ".join(MultivariateModel.names())}')
        self.end = endogenous
        self.ex = exogenous
        self.data = data
        self.kind = kind
        self.model = self.create_model(**kwargs)
        self._add_exogenous_on_fit = False
        self._add_endogenous_on_fit = False

    def create_model(self, **kwargs):
        if self.kind == Model.prophet.name:
            self._add_endogenous_on_fit = False
            model = Prophet(**kwargs)
            if self.ex:
                for ex in self.ex:
                    model.add_regressor(ex)

        elif self.kind == Model.holt_winters.name:
            data = self.data[self.end]
            model = ExponentialSmoothing(endog=data, **kwargs)

        elif self.kind == Model.arima.name:
            self._add_exogenous_on_fit = True
            self._add_endogenous_on_fit = True
            return AutoARIMA(**kwargs)

        elif self.kind == Model.varmax.name:
            data = self.data[self.end]
            ex = self.data[self.ex]
            return VARMAX(endog=data, exog=ex, **kwargs)

        elif self.kind == Model.sarimax.name:
            data = self.data[self.end]
            ex = self.data[self.ex]
            return SARIMAX(endog=data, exog=ex, **kwargs)
        else:
            raise
        return model

    def fit(self, **kwargs):
        if self._add_endogenous_on_fit and self._add_exogenous_on_fit:
            end = self.data[self.end]
            ex = self.data[self.ex]
            return self.model.fit(y=end, exogenous=ex, **kwargs)
        if self._add_exogenous_on_fit:
            return self.model.fit(self.data, **kwargs)
        else:
            return self.model.fit(**kwargs)
