from random import choice
from typing import Dict, List, Tuple

import numpy as np
import pandas

from hts._t import MethodT, NAryTreeT
from hts.hierarchy import make_iterable


def to_sum_mat(ntree: NAryTreeT) -> Tuple[np.ndarray, List[str]]:
    """
    This function creates a summing matrix for the bottom up and optimal combination approaches
    All the inputs are the same as above
    The output is a summing matrix, see Rob Hyndman's "Forecasting: principles and practice" Section 9.4

    Parameters
    ----------
    ntree : NAryTreeT


    Returns
    -------
    numpy.ndarray
        Summing matrix.

    List[str]
        Row order list of the level in the hierarchy represented by each row in the summing matrix.

    """
    nodes = ntree.level_order_traversal()
    node_labels = ntree.get_level_order_labels()
    num_at_level = list(map(sum, nodes))
    columns = num_at_level[-1]

    # Initialize summing matrix with bottom level rows
    sum_mat = np.identity(columns)

    # Names of each row in summing matrix.
    sum_mat_labels = []

    # Bottom level matrix labels, with indices correspoding to column in summing matrix
    bl_mat_idx_ref = node_labels[-1]

    # Skip total and bottom level of tree. Rows added outside of loop.
    for level in node_labels[1:-1]:
        for label in level:
            # Exclude duplicates specified in tree
            if label not in sum_mat_labels:
                row = []
                for bl_element in bl_mat_idx_ref:
                    # Check if the bottom level element is part of label
                    is_component = all(
                        [True if x in bl_element else False for x in label.split("_")]
                    )
                    if is_component:
                        row.append(1)
                    else:
                        row.append(0)

                # Add row correspoding to label to top of summing matrix
                row = np.array(row)
                sum_mat = np.vstack((row, sum_mat))
                sum_mat_labels.append(label)

    # Add top as first row in summing matrix
    top = np.ones(columns)
    sum_mat = np.vstack((top, sum_mat))

    # Reverse list of labels to match summing matrix, since vstack and append worked in the opposite order.
    # Not currently returned, but could be for information or matrix alignment.
    sum_mat_labels.reverse()
    sum_mat_labels = ["total"] + sum_mat_labels + bl_mat_idx_ref

    return sum_mat, sum_mat_labels


def project(
    hat_mat: np.ndarray, sum_mat: np.ndarray, optimal_mat: np.ndarray
) -> np.ndarray:
    new_mat = np.empty([hat_mat.shape[0], sum_mat.shape[0]])
    for i in range(hat_mat.shape[0]):
        new_mat[i, :] = np.dot(optimal_mat, np.transpose(hat_mat[i, :]))
    return new_mat


def y_hat_matrix(forecasts, keys=None):
    if not keys:
        keys = forecasts.keys()
    first = list(forecasts.keys())[0]
    y_hat_mat = np.zeros([len(forecasts[first].yhat), 1])
    for key in keys:
        f1 = np.array(forecasts[key].yhat)
        f2 = f1[:, np.newaxis]
        if np.all(y_hat_mat == 0):
            y_hat_mat = f2
        else:
            y_hat_mat = np.concatenate((y_hat_mat, f2), axis=1)
    return y_hat_mat


def optimal_combination(
    forecasts: Dict[str, pandas.DataFrame],
    sum_mat: np.ndarray,
    method: str,
    mse: Dict[str, float],
):
    """
    Produces the optimal combination of forecasts by trace minimization (as described by
    Wickramasuriya, Athanasopoulos, Hyndman in "Optimal Forecast Reconciliation for Hierarchical and Grouped Time
    Series Through Trace Minimization")

    Parameters
    ----------
    forecasts : dict
        Dictionary of pandas.DataFrames containing the future predictions
    sum_mat : np.ndarray
        The summing  matrix
    method : str
        One of:
            - OLS (ordinary least squares)
            - WLSS (structurally weighted least squares)
            - WLSV (variance weighted least squares)
    mse

    Returns
    -------

    """
    hat_mat = y_hat_matrix(forecasts)
    transpose = np.transpose(sum_mat)

    if method == MethodT.OLS.name:
        ols = np.dot(
            np.dot(sum_mat, np.linalg.inv(np.dot(transpose, sum_mat))), transpose
        )
        return project(hat_mat=hat_mat, sum_mat=sum_mat, optimal_mat=ols)
    elif method == MethodT.WLSS.name:
        diag = np.diag(np.transpose(np.sum(sum_mat, axis=1)))
    elif method == MethodT.WLSV.name:
        diag = [mse[key] for key in mse.keys()]
        diag = np.diag(np.flip(np.hstack(diag) + 0.0000001, 0))
    else:
        raise ValueError("Invalid method")

    # S*inv(S'S)*S'
    optimal_mat = np.dot(
        np.dot(
            np.dot(
                sum_mat,
                np.linalg.inv(np.dot(np.dot(transpose, np.linalg.inv(diag)), sum_mat)),
            ),
            transpose,
        ),
        np.linalg.inv(diag),
    )

    return project(hat_mat=hat_mat, sum_mat=sum_mat, optimal_mat=optimal_mat)


def proportions(nodes, forecasts, sum_mat, method=MethodT.PHA.name):
    n_cols = len(list(forecasts.keys()))
    fcst = forecasts[list(forecasts.keys())[0]].yhat
    fcst = fcst[:, np.newaxis]
    num_bts = sum_mat.shape[1]

    cols = [n.key for n in [nodes] + nodes.traversal_level()][
        (n_cols - num_bts) : n_cols
    ]

    bts_dat = nodes.to_pandas()[cols]
    if method == MethodT.AHP.name:
        divs = np.divide(np.transpose(np.array(bts_dat)), np.array(nodes.get_series()))
        props = divs.mean(1)
        props = props[:, np.newaxis]
    elif method == MethodT.PHA.name:
        bts_sum = bts_dat.sum(0)
        top_sum = sum(nodes.get_series())
        props = bts_sum / top_sum
        props = props[:, np.newaxis]
    else:
        raise ValueError("Invalid method")
    return np.dot(np.array(fcst), np.transpose(props))


def forecast_proportions(forecasts, nodes):
    """
    Cons:
       Produces biased revised forecasts even if base forecasts are unbiased
    """
    n_cols = len(list(forecasts.keys())) + 1

    levels = nodes.get_height()
    column = 0
    first_node = 1

    key = choice(list(forecasts.keys()))
    new_mat = np.empty([len(forecasts[key].yhat), n_cols - 1])
    new_mat[:, 0] = forecasts[key].yhat

    as_iterable = make_iterable(nodes, prop=None)

    for level in range(levels - 1):
        for i, node in enumerate(nodes.level_order_traversal()[level]):
            num_child = node
            last_node = first_node + num_child
            base_fcst = np.array(
                [forecasts[k.key].yhat[:] for k in as_iterable[first_node:last_node]]
            )
            print(base_fcst.shape)
            fore_sum = np.sum(base_fcst, axis=0)
            fore_sum = fore_sum[:, np.newaxis]
            if column == 0:
                rev_top = np.array(forecasts["total"].yhat)
                rev_top = rev_top[:, np.newaxis]
            else:
                rev_top = np.array(new_mat[:, column])
                rev_top = rev_top[:, np.newaxis]
            new_mat[:, first_node:last_node] = np.divide(
                np.multiply(np.transpose(base_fcst), rev_top), fore_sum
            )
            column += 1
            first_node += num_child
    return new_mat
