import numpy

from hts.hierarchy import HierarchyTree
from hts._t import Model
from hts.model import TimeSeriesModel


class HoltWintersModel(TimeSeriesModel):

    def __init__(self, node: HierarchyTree, **kwargs):
        super().__init__(Model.holt_winters.name, node, **kwargs)

    def predict(self, node: HierarchyTree,  steps_ahead=10):
        self.forecast = self._model.forecast(steps=steps_ahead).values
        in_sample_preds = self._model.predict(start=0, end=-1)

        self.residual = (in_sample_preds.values - self._reformat(self.node.item)[self.node.key].values)
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        return self.model

    def fit(self, **fit_args) -> 'TimeSeriesModel':
        self._model = self.model.fit(**fit_args)
        return self.model

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead)
