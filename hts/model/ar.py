import logging

import numpy

from hts.hierarchy import HierarchyTree
from hts._t import Model
from hts.model import TimeSeriesModel

logger = logging.getLogger(__name__)

try:
    from pmdarima import AutoARIMA
except ImportError:
    logger.error('pmdarima not installed, so HierarchicalArima won\'t work. Install it with: \npip install pmdarima')


class AutoArimaModel(TimeSeriesModel):

    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.auto_arima.name, node, **kwargs)

    def fit(self, **fit_args) -> 'TimeSeriesModel':
        as_df = self._reformat(self.node.item)
        end = as_df[self.node.key]
        if self.node.exogenous:
            ex = as_df[self.node.exogenous]
        else:
            ex = None
        self.model = self.model.fit(y=end, exogenous=ex, **fit_args)
        return self.model

    def predict(self, node, steps_ahead=10, alpha=0.05):
        if self.node.exogenous:
            ex = node.item
        else:
            ex = None
        self.forecast = self.model.predict(exogenous=ex, alpha=alpha, n_periods=steps_ahead)
        in_sample_preds = self.model.predict_in_sample()
        self.residual = in_sample_preds - self._reformat(self.node.item)[self.node.key].values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        return self.model

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, alpha=0.05, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead, alpha=alpha)


class SarimaxModel(TimeSeriesModel):
    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.sarimax.name, node, **kwargs)

    def fit(self, **fit_args) -> 'TimeSeriesModel':
        self.model = self.model.fit(disp=0, **fit_args)
        return self.model

    def predict(self, node, steps_ahead=10, alpha=0.05):
        if self.node.exogenous:
            ex = node.item
        else:
            ex = None
        self.forecast = self.model.forecast(steps=steps_ahead, exog=ex).values
        in_sample_preds = self.model.get_prediction(dynamic=False, exog=ex)
        self.residual = (in_sample_preds.predicted_mean - self._reformat(self.node.item)[self.node.key].values).values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        return self.model

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, alpha=0.05, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead, alpha=alpha)
