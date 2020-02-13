import numpy

from hts.hierarchy import HierarchyTree
from hts._t import Model
from hts.model import TimeSeriesModel


class HoltWintersModel(TimeSeriesModel):
    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.holt_winters.name, node, **kwargs)

    def fit_predict(self, start=None, end=None, **fit_args):
        self.model.fit(**fit_args)
        self.forecast = self.model.predict(start=None, end=None)
        self.residual = end - self.forecast.yhat[:-periods].values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)