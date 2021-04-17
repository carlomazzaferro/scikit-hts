import numpy

from hts._t import MethodT
from hts.core.exceptions import InvalidArgumentException
from hts.functions import (
    forecast_proportions,
    optimal_combination,
    proportions,
    y_hat_matrix,
)
from hts.hierarchy.utils import make_iterable


class RevisionMethod(object):
    def __init__(
        self,
        name: str,
        sum_mat: numpy.ndarray,
        transformer,
    ):
        self.name = name
        self.transformer = transformer
        self.sum_mat = sum_mat

    def _new_mat(self, y_hat_mat) -> numpy.ndarray:
        new_mat = numpy.empty([y_hat_mat.shape[0], self.sum_mat.shape[0]])
        for i in range(y_hat_mat.shape[0]):
            new_mat[i, :] = numpy.dot(self.sum_mat, numpy.transpose(y_hat_mat[i, :]))
        return new_mat

    def _y_hat_matrix(self, forecasts) -> numpy.ndarray:
        n_cols = len(list(forecasts.keys())) + 1
        if self.name == "BU":
            keys = list(forecasts.keys())[
                n_cols - self.sum_mat.shape[1] - 1 : n_cols - 1
            ]
        else:
            keys = range(n_cols - self.sum_mat.shape[1] - 1, n_cols - 1)
        return y_hat_matrix(forecasts, keys=keys)

    def revise(self, forecasts=None, mse=None, nodes=None) -> numpy.ndarray:
        """


        Parameters
        ----------
        forecasts
        mse
        nodes

        Returns
        -------

        """
        if self.name == MethodT.NONE.name:
            return y_hat_matrix(forecasts=forecasts)

        if self.name in [MethodT.OLS.name, MethodT.WLSS.name, MethodT.WLSV.name]:
            return optimal_combination(
                forecasts=forecasts, sum_mat=self.sum_mat, method=self.name, mse=mse
            )

        elif self.name == MethodT.BU.name:
            y_hat = self._y_hat_matrix(forecasts)
            return self._new_mat(y_hat)

        elif self.name in [MethodT.AHP.name, MethodT.PHA.name]:
            if self.transformer:
                for node in make_iterable(nodes, prop=None):
                    node.item[node.key] = self.transformer.inverse_transform(
                        node.item[node.key]
                    )
            y_hat = proportions(
                nodes=nodes, forecasts=forecasts, sum_mat=self.sum_mat, method=self.name
            )
            return self._new_mat(y_hat)

        elif self.name == MethodT.FP.name:
            return forecast_proportions(forecasts, nodes)

        else:
            raise InvalidArgumentException("Revision model name is invalid")
