import logging

import numpy
import pandas

from hts.hierarchy import HierarchyTree
from hts._t import Model
from hts.model.base import TimeSeriesModel
from hts.utilities.utils import suppress_stdout_stderr

logger = logging.getLogger(__name__)

logging.getLogger('fbprophet').setLevel(logging.ERROR)


class FBProphetModel(TimeSeriesModel):
    """
    Wrapper class around ``fbprophet.Prophet``

    Attributes
    ----------
    model : Prophet
        The instance of the model

    mse : float
        MSE for in-sample predictions

    residual : numpy.ndarry
        Residuals for the in-sample predictions

    forecast : pandas.DataFramer
        The forecast for the trained model

    Methods
    -------
    fit(self, **fit_args)
        Fits underlying models to the data, passes kwargs to ``SARIMAX``

    predict(self, node, steps_ahead: int = 10, freq: str = 'D', **predict_args)
        Predicts the n-step ahead forecast. Exogenous variables are required if models were
        fit using them, frequency should be passed as well
    """
    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.prophet.name, node, **kwargs)
        self.cap = None
        self.floor = None
        self.include_history = False

    def create_model(self,
                     capacity_max=None,
                     capacity_min=None,
                     **kwargs):
        self.cap = capacity_max
        self.floor = capacity_min

        if not capacity_max and not capacity_min:
            growth = 'linear'
        else:
            growth = 'logistic'
            if self.cap:
                self.node.item['cap'] = capacity_max
            if self.floor:
                self.node.item['floor'] = capacity_min

        try:
            from fbprophet import Prophet
        except ImportError:  # pragma: no cover
            logger.error('prophet model requires fbprophet to work. Exiting.'
                         'Install it with: pip install scikit-hts[prophet]')
            return
        model = Prophet(growth=growth, **kwargs)
        if self.node.exogenous:
            for ex in self.node.exogenous:
                model.add_regressor(ex)
        return model

    def _reformat(self, node):
        if isinstance(node, pandas.Series):
            node = pandas.DataFrame(node)
        df = node.rename(columns={self.node.key: 'y'})
        df['ds'] = pandas.to_datetime(df.index)
        return df.reset_index(drop=True)

    def fit(self, **fit_args) -> 'TimeSeriesModel':
        df = self._reformat(self.node.item)
        with suppress_stdout_stderr():
            self.model = self.model.fit(df)
            self.model.stan_backend = None
        return self

    def predict(self,
                node: HierarchyTree,
                freq: str = 'D',
                steps_ahead: int = 1):

        df = self._reformat(node.item)
        future = self.model.make_future_dataframe(periods=steps_ahead,
                                                  freq=freq,
                                                  include_history=True)
        if self.cap:
            future['cap'] = self.cap
        if self.floor:
            future['floor'] = self.floor

        self.forecast = self.model.predict(future)
        merged = pandas.merge(df, self.forecast, on='ds')
        self.residual = (merged['yhat'] - merged['y']).values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        if self.cap is not None:
            self.forecast.yhat = numpy.exp(self.forecast.yhat)
        if self.transform:
            self.forecast.yhat = self.transformer.inverse_transform(self.forecast.yhat)
            self.forecast.trend = self.transformer.inverse_transform(self.forecast.trend)
            for component in ["seasonal", "daily", "weekly", "yearly", "holidays"]:
                if component in self.forecast.columns.tolist():
                    inv_transf = self.transformer.inverse_transform(getattr(self.forecast, component))
                    setattr(self.forecast, component, inv_transf)
        return self
