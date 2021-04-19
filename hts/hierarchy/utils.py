import logging
from typing import List, Optional, Tuple

import pandas

from hts._t import NAryTreeT
from hts.core.exceptions import InvalidArgumentException

logger = logging.getLogger(__name__)


def make_iterable(nodes: NAryTreeT, prop="key"):
    if not prop:
        it = [nodes] + nodes.traversal_level()
    else:
        it = [getattr(nodes, prop)] + [
            getattr(k, prop) for k in nodes.traversal_level()
        ]
    return it


def fetch_cols(exogenous, name) -> Tuple[List[str], Optional[List[str]]]:
    if not exogenous:
        return [name], None
    exog = exogenous.get(name, None)
    cols = [name] + exog if exog else [name]
    return cols, exog


def hexify(df, lat_col, lon_col, levels=(6, 8)) -> Optional[pandas.DataFrame]:
    try:
        from h3 import h3
    except ImportError:  # pragma: no cover
        logger.error(
            "h3-py must be installed for geo hashing capabilities. Exiting."
            "Install it with: pip install scikit-hts[geo]"
        )
        return
    for r in range(levels[0], levels[1] + 1):
        df[f"hex_index_{r}"] = df.apply(
            lambda x: h3.geo_to_h3(x[lat_col], x[lon_col], r), 1
        )
    return df


def resample_count(df: pandas.DataFrame, freq: str, colname: str) -> pandas.DataFrame:
    _df = pandas.DataFrame({colname: df.index}, index=df.index)
    return _df.resample(freq).agg("count")


def groupify(
    root_node, df, freq="1H", nodes=None, min_count=0.1, total=None
) -> NAryTreeT:
    """

    Parameters
    ----------
    fillna
    root_node : NAryTreeT
    df : pandas.DataFrame
    freq : str
        resample frequency
    nodes : tuple
        Hierarchy node
    min_count : int or float
        Minimum number of observations for a node to be used. If float, it represents a
        fraction of values that need to be non NaNs and must be between 0 and 1
    total : pandas.DataFrame:
        Name of root node

    Returns
    -------

    """

    child_group = nodes[0]  # city
    children = df[child_group].unique()  # berlin, munich, ...

    # add first level children
    for child in children:
        sub_df = df[df[child_group] == child]
        resampled = resample_count(sub_df, freq, child)
        root_node.add_child(key=child, item=resampled, exogenous=None)

    # add the rest
    for node in nodes[1:]:
        parent_group = child_group
        child_group = node  # hex_index_6
        children = df[child_group].unique()  # abcccc, abccf, ...

        for child in children:
            sub_df = df[df[child_group] == child]
            if isinstance(min_count, float):
                allowance = len(total) * min_count
            elif isinstance(min_count, int):
                allowance = min_count
            else:
                raise InvalidArgumentException(
                    "min_count must be either float or integer"
                )
            if len(sub_df) < allowance:
                continue
            parent_name = sub_df[parent_group].value_counts().index[0]
            resampled = resample_count(sub_df, freq, child)
            for c in root_node.traversal_level():
                if c.key == parent_name:
                    c.add_child(key=child, item=resampled, exogenous=None)
    return root_node
