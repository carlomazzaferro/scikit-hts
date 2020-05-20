import logging
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning

from hts.hierarchy import HierarchyTree
from hts._t import Model
from hts.model.base import TimeSeriesModel


class AutoArimaModel(TimeSeriesModel):
    """
    Wrapper class around ``pmdarima.AutoARIMA``

    Attributes
    ----------
    model : pmdarima.AutoARIMA
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
        Fits underlying models to the data, passes kwargs to ``AutoARIMA``

    predict(self, node, steps_ahead: int = 10, alpha: float = 0.05)
        Predicts the n-step ahead forecast. Exogenous variables are required if models were
        fit using them
    """
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
        return self

    def predict(self, node, steps_ahead=10, alpha=0.05):
        if self.node.exogenous:
            ex = node.item
        else:
            ex = None
        y_hat = self.model.predict(exogenous=ex, alpha=alpha, n_periods=steps_ahead)
        in_sample_preds = self.model.predict_in_sample()
        return self._set_results_return_self(in_sample_preds, y_hat)

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, alpha=0.05, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead, alpha=alpha)


class SarimaxModel(TimeSeriesModel):
    """
    Wrapper class around ``statsmodels.tsa.statespace.sarimax.SARIMAX``

    Attributes
    ----------
    model : SARIMAX
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

    predict(self, node, steps_ahead: int = 10, alpha: float = 0.05)
        Predicts the n-step ahead forecast. Exogenous variables are required if models were
        fit using them
    """
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
        y_hat = self.model.forecast(steps=steps_ahead, exog=ex).values
        in_sample_preds = self.model.get_prediction(dynamic=False, exog=ex).predicted_mean
        return self._set_results_return_self(in_sample_preds, y_hat)

    def fit_predict(self, node: HierarchyTree, steps_ahead=10, alpha=0.05, **fit_args):
        return self.fit(**fit_args).predict(node=node, steps_ahead=steps_ahead, alpha=alpha)
