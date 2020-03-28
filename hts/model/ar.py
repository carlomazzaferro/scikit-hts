import logging
import warnings

import numpy
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from hts.hierarchy import HierarchyTree
from hts._t import Model
from hts.model.base import TimeSeriesModel

logger = logging.getLogger(__name__)

try:
    from pmdarima import AutoARIMA
except ImportError:
    logger.warning('pmdarima not installed, so auto_arima won\'t work. Install it with: \npip install pmdarima')


class AutoArimaModel(TimeSeriesModel):

    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.auto_arima.name, node, **kwargs)

    def fit(self, **fit_args) -> 'TimeSeriesModel':
        as_df = self.node.item
        end = self.node.get_series()
        if self.node.exogenous:
            ex = as_df[self.node.exogenous]
        else:
            ex = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model = self.model.fit(y=end, exogenous=ex, **fit_args)
        return self.model

    def predict(self, node, steps_ahead=10, alpha=0.05):
        if self.node.exogenous:
            ex = node.item
        else:
            ex = None
        self.forecast = self.model.predict(exogenous=ex, alpha=alpha, n_periods=steps_ahead)
        in_sample_preds = self.model.predict_in_sample()
        self.residual = (in_sample_preds - self.node.get_series()).values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        return self.model

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, alpha=0.05, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead, alpha=alpha)


class SarimaxModel(TimeSeriesModel):
    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.sarimax.name, node, **kwargs)

    def fit(self, **fit_args) -> 'TimeSeriesModel':
        self.model = self.model.fit(disp=0, **fit_args)
        return self

    def predict(self, node, steps_ahead=10, alpha=0.05):
        if self.node.exogenous:
            ex = node.item
        else:
            ex = None
        self.forecast = self.model.forecast(steps=steps_ahead, exog=ex).values
        in_sample_preds = self.model.get_prediction(dynamic=False, exog=ex)
        self.residual = (in_sample_preds.predicted_mean - self.node.get_series()).values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        return self

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, alpha=0.05, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead, alpha=alpha)
