from typing import Dict, Optional

import numpy
import pandas

from hts._t import MethodsT, NAryTreeT, TransformT
from hts.functions import to_sum_mat
from hts.revision import RevisionMethod


def revise_forecasts(method: str,
                     forecasts: Dict[str, numpy.ndarray],
                     errors: Optional[Dict[str, numpy.ndarray]] = None,
                     residuals: Optional[Dict[str, numpy.ndarray]] = None,
                     summing_matrix: numpy.ndarray = None,
                     nodes: NAryTreeT = None,
                     transformer: TransformT = None
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
        summing_matrix = to_sum_mat(nodes)

    if method in [MethodsT.AHP.name, MethodsT.PHA.name, MethodsT.FP.name] and not nodes:
        raise ValueError(f'Method {method} requires an NAryTree to be passed')

    if method in [MethodsT.OLS.name, MethodsT.WLSS.name, MethodsT.WLSV.name]:
        if not (all([forecasts, errors, residuals]) or (not summing_matrix)):
            raise ValueError(f'Method {method} requires forecasts, errors, and residuals to be passed, as '
                             f'well as an NAryTree or a summing matrix')

    revision = RevisionMethod(
        name=method,
        sum_mat=summing_matrix,
        transformer=transformer
    )

    revised = revision.revise(forecasts=forecasts,
                              mse=errors,
                              nodes=nodes)

    return pandas.DataFrame(revised, columns=list(forecasts.keys()))
