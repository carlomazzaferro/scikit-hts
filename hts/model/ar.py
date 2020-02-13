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

    def fit_predict(self, periods=10, **fit_args):
        end = self.node.item[self.node.key]
        ex = self.node.item[self.node.exogenous]
        self.forecast = self.model.fit_predict(y=end, exogenous=ex, n_periods=periods, **fit_args)

        self.residual = end - self.forecast.yhat[:-periods].values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)


class SarimaxModel(TimeSeriesModel):
    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.sarimax.name, node, **kwargs)

    def fit_predict(self, start=None, end=None, dynamic=False, index=None,
                    exog=None, **fit_args):
        # TODO: figure out how to pass exog features here
        self.model = self.model.fit(**fit_args)
        self.forecast = self.model.get_prediction(start=start,
                                                  end=end,
                                                  dynamic=dynamic, index=index,
                                                  exog=exog)

        self.residual = end - self.forecast.yhat[:-periods].values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
