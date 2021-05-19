from typing import Dict, Optional, Union

import numpy
import pandas

from hts._t import MethodT, NAryTreeT, TransformT
from hts.functions import to_sum_mat
from hts.revision import RevisionMethod


def _sanitize_forecasts_dict(
    forecasts: Dict[str, Union[numpy.ndarray, pandas.Series, pandas.DataFrame]]
):
    for k, v in forecasts.items():
        if isinstance(v, numpy.ndarray):
            if v.ndim > 1:
                raise ValueError("Forecasts must be of a dimension 1")
            forecasts[k] = pandas.DataFrame({"yhat": v})
        elif isinstance(v, pandas.Series):
            forecasts[k] = pandas.DataFrame({"yhat": v})
        elif isinstance(v, pandas.DataFrame):
            if len(v.columns) > 1:
                raise ValueError(
                    "If providing forecasts as a DataFrame, it must have one column only"
                )
            col_name = v.columns[0]
            v.rename(columns={col_name: "yhat"}, inplace=True)
            forecasts[k] = v
        else:
            raise ValueError(
                "`forcasts` must be a dict mapping string to array, series or DataFrame"
            )
    return forecasts


def revise_forecasts(
    method: str,
    forecasts: Dict[str, numpy.ndarray],
    errors: Optional[Dict[str, numpy.ndarray]] = None,
    residuals: Optional[Dict[str, numpy.ndarray]] = None,
    summing_matrix: numpy.ndarray = None,
    nodes: NAryTreeT = None,
    transformer: TransformT = None,
):
    """
    Convenience function to get revised forecast for pre-computed base forecasts

    Parameters
    ----------
    method : str
        The reconciliation method to use
    forecasts : Dict[str, numpy.ndarray]
        A dict mapping key name to its forecasts (including in-sample forecasts). Required.
    errors : Dict[str, numpy.ndarray]
        A dict mapping key name to the in-sample errors. Required for methods: ``OLS``, ``WLSS``, ``WLSV``
    residuals : Dict[str, numpy.ndarray]
        A dict mapping key name to the residuals of in-sample forecasts. Required for methods: OLS, WLSS, WLSV
    summing_matrix : numpy.ndarray
        Not required if ``nodes`` argument is passed, or if using ``BU`` approach
    nodes : NAryTreeT
        The tree of nodes as specified in :py:class:`HierarchyTree <hts.hierarchy.HierarchyTree>`. Required if not
        if using ``AHP``, ``PHA` ``FP`` methods, or if using  passing the ``OLS``, ``WLSS``, ``WLSV`` methods
          and not passing the ``summing_matrix`` parameter
    transformer : TransformT
        A transform with the method: ``inv_func`` that will be applied to the forecasts

    Returns
    -------
    revised forecasts : pandas.DataFrame
        The revised forecasts
    """

    if nodes:
        summing_matrix, sum_mat_labels = to_sum_mat(nodes)

    if method in [MethodT.AHP.name, MethodT.PHA.name, MethodT.FP.name] and not nodes:
        raise ValueError(f"Method {method} requires an NAryTree to be passed")

    if method in [MethodT.OLS.name, MethodT.WLSS.name, MethodT.WLSV.name]:
        if not (all([forecasts, errors, residuals]) or (not summing_matrix)):
            raise ValueError(
                f"Method {method} requires forecasts, errors, and residuals to be passed, as "
                f"well as an NAryTree or a summing matrix"
            )

    revision = RevisionMethod(
        name=method, sum_mat=summing_matrix, transformer=transformer
    )
    sanitized_forecasts = _sanitize_forecasts_dict(forecasts)
    revised = revision.revise(forecasts=sanitized_forecasts, mse=errors, nodes=nodes)

    return pandas.DataFrame(revised, columns=list(sanitized_forecasts.keys()))
