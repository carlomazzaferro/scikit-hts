from typing import Dict, Optional, Union

import numpy
import pandas

from hts._t import ArrayLike, MethodT, NAryTreeT, TransformT
from hts.functions import to_sum_mat
from hts.revision import RevisionMethod


def _to_numpy(v: ArrayLike, kind: str = "forecasts") -> numpy.ndarray:
    if isinstance(v, numpy.ndarray):
        if v.ndim > 1:
            raise ValueError(f"`{kind}` values must be of a dimension 1")
        return v
    elif isinstance(v, pandas.DataFrame):
        if len(v.columns) > 1:
            raise ValueError(
                f"If providing `{kind}` as a DataFrame, it must have one column only"
            )
        col_name = v.columns[0]
        return v[col_name].values
    elif isinstance(v, pandas.Series):
        return v.values
    else:
        raise ValueError(
            f"`{kind}` must be a dict mapping string to array, series or DataFrame"
        )


def _sanitize_errors_dict(errors: Dict[str, float]) -> Dict[str, float]:
    for k, v in errors.items():
        if not isinstance(v, float):
            raise ValueError("`errors` dict must be a mapping from string to float")
    return errors


def _sanitize_residuals_dict(
    residuals: Dict[str, ArrayLike]
) -> Dict[str, numpy.ndarray]:
    for k, v in residuals.items():
        residuals[k] = _to_numpy(v, kind="residuals")
    return residuals


def _sanitize_forecasts_dict(
    forecasts: Dict[str, ArrayLike]
) -> Dict[str, pandas.DataFrame]:

    for k, v in forecasts.items():
        as_array = _to_numpy(v, kind="forecasts")
        forecasts[k] = pandas.DataFrame({"yhat": as_array})
    return forecasts


def _calculate_errors(
    method: str,
    errors: Optional[Dict[str, float]] = None,
    residuals: Optional[Dict[str, numpy.ndarray]] = None,
):
    errors_or_residuals = (
        True if (errors is not None or residuals is not None) else False
    )
    if not errors_or_residuals:
        raise ValueError(
            f"Method {method} requires either errors or residuals to be provided"
        )
    if residuals is not None:
        residuals = _sanitize_residuals_dict(residuals)
        if errors is None:
            errors = {}
        for k, v in residuals.items():
            errors[k] = numpy.mean(numpy.array(v) ** 2)
    return _sanitize_errors_dict(errors)


def revise_forecasts(
    method: str,
    forecasts: Dict[str, ArrayLike],
    errors: Optional[Dict[str, float]] = None,
    residuals: Optional[Dict[str, ArrayLike]] = None,
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
    forecasts : Dict[str, ArrayLike]
        A dict mapping key name to its forecasts (including in-sample forecasts). Required, can be
        of type ``numpy.ndarray`` of ``ndim == 1``, ``pandas.Series``, or single columned ``pandas.DataFrame``
    errors : Dict[str, float]
        A dict mapping key name to the in-sample errors. Required for methods: ``OLS``, ``WLSS``, ``WLSV`` if
        ``residuals`` is not passed
    residuals : Dict[str, ArrayLike]
        A dict mapping key name to the residuals of in-sample forecasts. Required for methods: ``OLS``, ``WLSS``,
        ``WLSV``, can be of type ``numpy.ndarray`` of ndim == 1, ``pandas.Series``, or single columned
        ``pandas.DataFrame``. If passing residuals, ``errors`` dict is not required and will instead be calculated
        using MSE metric: ``numpy.mean(numpy.array(residual) ** 2)``
    summing_matrix : numpy.ndarray
        Not required if ``nodes`` argument is passed, or if using ``BU`` approach
    nodes : NAryTreeT
        The tree of nodes as specified in :py:class:`HierarchyTree <hts.hierarchy.HierarchyTree>`. Required if not
        if using ``AHP``, ``PHA``, ``FP`` methods, or if using  passing the ``OLS``, ``WLSS``, ``WLSV`` methods
        and not passing the ``summing_matrix`` parameter
    transformer : TransformT
        A transform with the method: ``inv_func`` that will be applied to the forecasts

    Returns
    -------
    revised forecasts : ``pandas.DataFrame``
        The revised forecasts
    """

    if nodes:
        summing_matrix, sum_mat_labels = to_sum_mat(nodes)

    if method in [MethodT.AHP.name, MethodT.PHA.name, MethodT.FP.name] and not nodes:
        raise ValueError(f"Method {method} requires an NAryTree to be passed")

    if method in [MethodT.OLS.name, MethodT.WLSS.name, MethodT.WLSV.name]:
        errors = _calculate_errors(method=method, errors=errors, residuals=residuals)
        if not (all([forecasts, errors]) or (not summing_matrix)):
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
