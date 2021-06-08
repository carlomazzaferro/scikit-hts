import logging
from typing import NamedTuple, Union

import numpy
import pandas
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from hts._t import ModelT, NAryTreeT, TimeSeriesModelT, TransformT
from hts.core.exceptions import InvalidArgumentException
from hts.hierarchy import HierarchyTree
from hts.transforms import BoxCoxTransformer, FunctionTransformer

logger = logging.getLogger(__name__)


class TimeSeriesModel(TimeSeriesModelT):
    """Base class for the implementation of the underlying models.
    Inherits from scikit-learn base classes
    """

    def __init__(
        self, kind: str, node: HierarchyTree, transform: TransformT = False, **kwargs
    ):
        """
        Parameters
        ----------
        kind : str
            One of `prophet`, `sarimax`, `auto-arima`, `holt-winters`
        node : HierarchyTree
            Node
        transform : Bool or NamedTuple
        kwargs
            Keyword arguments to be passed to the model instantiation. See the documentation
            of each of the actual model implementations for a more comprehensive treatment
        """

        if kind not in ModelT.names():
            raise InvalidArgumentException(
                f'Model {kind} not valid. Pick one of: {" ".join(ModelT.names())}'
            )

        self.kind = kind
        self.node = node
        self.transform_function = self._set_transform(transform=transform)
        self.model = self.create_model(**kwargs)
        self.forecast = None
        self.residual = None
        self.mse = None

    def _set_transform(self, transform: TransformT):
        if transform is False or transform is None:
            return FunctionTransformer(func=self._no_func, inv_func=self._no_func)
        elif transform is True:
            return BoxCoxTransformer()
        elif isinstance(transform, tuple):
            if not hasattr(transform, "func") or not hasattr(transform, "inv_func"):
                raise ValueError(
                    "If passing a NamedTuple, it must have a `func` and `inv_func` parameters"
                )
            return FunctionTransformer(
                func=getattr(transform, "func"), inv_func=getattr(transform, "inv_func")
            )
        else:
            raise ValueError(
                "Invalid transform passed. Use either `True` for default boxcox transform or "
                "a `NamedTuple(func: Callable, inv_func: Callable)` for custom transforms"
            )

    def _set_results_return_self(self, in_sample, y_hat):
        in_sample = self.transform_function.inverse_transform(in_sample)
        y_hat = self.transform_function.inverse_transform(y_hat)
        self.forecast = pandas.DataFrame(
            {"yhat": numpy.concatenate([in_sample, y_hat])}
        )
        self.residual = (in_sample - self._get_transformed_data(as_series=True)).values
        self.mse = numpy.mean(numpy.array(self.residual) ** 2)
        return self

    def _get_transformed_data(
        self, as_series: bool = False
    ) -> Union[pandas.DataFrame, pandas.Series]:
        key = self.node.key
        value = self.node.item
        transformed = self.transform_function.transform(value[key])
        if as_series:
            return pandas.Series(transformed)
        else:
            return pandas.DataFrame({key: transformed})

    def create_model(self, **kwargs):

        if self.kind == ModelT.holt_winters.name:
            data = self._get_transformed_data()
            model = ExponentialSmoothing(endog=data, **kwargs)

        elif self.kind == ModelT.auto_arima.name:
            try:
                from pmdarima import AutoARIMA
            except ImportError:  # pragma: no cover
                logger.error(
                    "pmdarima not installed, so auto_arima won't work. Exiting."
                    "Install it with: pip install scikit-hts[auto_arima]"
                )
                return
            model = AutoARIMA(**kwargs)

        elif self.kind == ModelT.sarimax.name:
            as_df = self.node.item
            end = self._get_transformed_data(as_series=True)
            if self.node.exogenous:
                ex = as_df[self.node.exogenous]
            else:
                ex = None
            model = SARIMAX(endog=end, exog=ex, **kwargs)
        else:
            raise
        return model

    def fit(self, **fit_args) -> "TimeSeriesModel":
        raise NotImplementedError

    def predict(self, node: HierarchyTree, **predict_args):
        raise NotImplementedError

    def fit_predict(self, node: HierarchyTree, **kwargs):
        return self.fit().predict(node)
