from random import choice
from typing import Dict, List, Tuple

import numpy as np
import pandas

from hts._t import MethodT, NAryTreeT
from hts.hierarchy import make_iterable


def to_sum_mat(
    ntree: NAryTreeT = None, node_labels: List[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    This function creates a summing matrix for the bottom up and optimal combination approaches
    All the inputs are the same as above
    The output is a summing matrix, see Rob Hyndman's "Forecasting: principles and practice" Section 9.4

    Parameters
    ----------
    ntree : NAryTreeT

    node_labels : List[str]
        Labels corresponing to node names/ summing matrix. Get from hts.functions.get_hierarchichal_df(...)

    Returns
    -------
    numpy.ndarray
        Summing matrix.

    List[str]
        Row order list of the level in the hierarchy represented by each row in the summing matrix.

    """
    if node_labels:
        columns = len(node_labels[-1])
    elif ntree:
        nodes = ntree.level_order_traversal()
        node_labels = ntree.get_level_order_labels()
        num_at_level = list(map(sum, nodes))
        columns = num_at_level[-1]
    else:
        raise ValueError(
            "Must pass either ntree or node_labels to the function. Neither was received."
        )

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


def get_agg_series(df: pandas.DataFrame, levels: List[List[str]]) -> List[str]:
    """
    Get aggregate level series names.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabular data.
    levels : List[List[str]]
        List of lists containing the desired level of aggregation.

    Returns
    -------
    List[str]
        Aggregate series names.
    """
    grouped_levels = []
    for level in levels:
        cross_vals = list(
            "_".join(x for x in y) for y in df[level].drop_duplicates().values
        )
        grouped_levels += cross_vals
    return grouped_levels


def _create_bl_str_col(df: pandas.DataFrame, level_names: List[str]) -> List[str]:
    """
    Concatenate the column values of all the specified level_names by row into a single column.

    Parameters
    ----------
    df : pandas.DataFrame
        Tabular data.
    level_names : List[str]
        Levels in the hierarchy.

    Returns
    -------
    List[str]
        Concatendated column values by row.
    """
    return list("_".join(x for x in y) for y in df[level_names].values)


def add_agg_series_to_df(
    df: pandas.DataFrame, grouped_levels: List[str], bottom_levels: List[str]
) -> pandas.DataFrame:
    """
    Add aggregate series columns to wide dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide dataframe containing bottom level series.
    grouped_levels : List[str]
        Grouped level, underscore delimited, column names.
    bottom_levels : List[str]
        Bottom level, underscore delimited, column names.

    Returns
    -------
    pandas.DataFrame
        Wide dataframe with all series in hierarchy.
    """
    component_cols = _get_bl(grouped_levels, bottom_levels)
    # Add series as specified grouping levels
    for i, cols in enumerate(component_cols):
        df[grouped_levels[i]] = df[cols].sum(axis=1)
    return df


def _get_bl(grouped_levels: List[str], bottom_levels: List[str]) -> List[List[str]]:
    """
    Get bottom level columns required to sum to create grouped columns.

    Parameters
    ----------
    grouped_levels : List[str]
        Grouped level, underscore delimited, column names.
    bottom_levels : List[str]
        Bottom level, underscore delimited, column names.

    Returns
    -------
    List[List[str]]
        Bottom level column names that make up each individual aggregated node in the hierarchy.
    """
    # Split groupings by "_" b/c this makes it possible to search column names
    grouped_levels_split = [lev.split("_") for lev in grouped_levels]
    bottom_levels_split = [lev.split("_") for lev in bottom_levels]

    cols_to_add = []
    for lev in grouped_levels_split:
        group_bl_cols = [
            bl_col for bl_col in bottom_levels_split if set(lev).issubset(bl_col)
        ]
        cols_to_add.append(["_".join(lev) for lev in group_bl_cols])
    return cols_to_add


def get_hierarchichal_df(
    df: pandas.DataFrame,
    level_names: List[str],
    hierarchy: List[List[str]],
    date_colname: str,
    val_colname: str,
) -> Tuple[pandas.DataFrame, np.array, List[str]]:
    """
    Transform your tabular dataframe to a wide dataframe with desired levels a hierarchy.

    Parameters
    ----------
    df : pd.DataFrame
        Tabular dataframe
    level_names : List[str]
        Levels in the hierarchy.
    hierarchy : List[List[str]]
        Desired levels in your hierarchy.
    date_colname : str
        Date column name
    val_colname : str
        Name of column containing series values.

    Returns
    -------
    pd.DataFrame
        Wide dataframe with levels of specified aggregation.

    np.array
        Summing matrix.

    List[str]:
        Summing matrix labels.

    Examples
    --------
    >>> import hts.functions
    >>> hier_df = pandas.DataFrame(
        data={
            'ds': ['2020-01', '2020-02'] * 5,
            "lev1": ['A', 'A',
                     'A', 'A',
                     'A', 'A',
                     'B', 'B',
                     'B', 'B'],
            "lev2": ['X', 'X',
                     'Y', 'Y',
                     'Z', 'Z',
                     'X', 'X',
                     'Y', 'Y'],
            "val": [1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10]
        }
    )
    >>> hier_df
            ds lev1 lev2  val
    0  2020-01    A    X    1
    1  2020-02    A    X    2
    2  2020-01    A    Y    3
    3  2020-02    A    Y    4
    4  2020-01    A    Z    5
    5  2020-02    A    Z    6
    6  2020-01    B    X    7
    7  2020-02    B    X    8
    8  2020-01    B    Y    9
    9  2020-02    B    Y   10
    >>> level_names = ['lev1', 'lev2']
    >>> hierarchy = [['lev1'], ['lev2']]
    >>> wide_df, sum_mat, sum_mat_labels = hts.functions.get_hierarchichal_df(hier_df,
                                                                              level_names=level_names,
                                                                              hierarchy=hierarchy,
                                                                              date_colname='ds',
                                                                              val_colname='val')
    >>> wide_df
        lev1_lev2  A_X  A_Y  A_Z  B_X  B_Y  total   A   B   X   Y  Z
        ds
        2020-01      1    3    5    7    9     25   9  16   8  12  5
        2020-02      2    4    6    8   10     30  12  18  10  14  6
    """
    # Column names separated by underscores
    level_names_underscores = "_".join(level_names)

    # Create a column representing the bottom level of aggregation
    df[level_names_underscores] = _create_bl_str_col(df, level_names)

    # Pivot df to bottom level. We can create the aggregate these series to get all the higher levels.
    forecast_df = df.pivot(
        index=date_colname, columns=level_names_underscores, values=val_colname
    )

    # Sum all bottom level series to get total
    forecast_df["total"] = forecast_df.sum(axis=1)

    bottom_levels = list(df[level_names_underscores].unique())

    grouped_levels = get_agg_series(df, hierarchy)

    sum_mat, sum_mat_labels = to_sum_mat(
        ntree=None, node_labels=[["total"], grouped_levels, bottom_levels]
    )

    forecast_df = add_agg_series_to_df(forecast_df, grouped_levels, bottom_levels)

    return forecast_df, sum_mat, sum_mat_labels
