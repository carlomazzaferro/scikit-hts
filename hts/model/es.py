from hts._t import ModelT
from hts.hierarchy import HierarchyTree
from hts.model.base import TimeSeriesModel


class HoltWintersModel(TimeSeriesModel):
    """
    Wrapper class around ``statsmodels.tsa.holtwinters.ExponentialSmoothing``

    Attributes
    ----------
    model : ExponentialSmoothing
        The instance of the model

    _model : HoltWintersResults
        The result of model fitting. See statsmodels.tsa.holtwinters.HoltWintersResults

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

    predict(self, node, steps_ahead: int = 10)
        Predicts the n-step ahead forecast
    """

    def __init__(self, node: HierarchyTree, **kwargs):
        self._model = None
        super().__init__(ModelT.holt_winters.name, node, **kwargs)

    def predict(self, node: HierarchyTree, steps_ahead=10):
        y_hat = self._model.forecast(steps=steps_ahead).values
        in_sample_preds = self._model.predict(start=0, end=-1).values
        return self._set_results_return_self(in_sample_preds, y_hat)

    def fit(self, **fit_args) -> "TimeSeriesModel":
        self._model = self.model.fit(**fit_args)
        return self

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead)
