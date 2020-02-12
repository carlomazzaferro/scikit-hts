import pandas
from h3 import h3

from hts import HierarchyTree
from hts._t import NAryTreeT
from hts.exceptions import InvalidArgumentException


def fetch_cols(exogenous, name):
    if not exogenous:
        return name
    exog = exogenous.get(name, None)
    cols = [name] + exog if exog else name
    return cols, exog


def hexify(df, lat_col, lon_col, levels=(6, 8)):
    for r in range(levels[0], levels[1] + 1):
        df[f'hex_index_{r}'] = df.apply(lambda x: h3.geo_to_h3(x[lat_col], x[lon_col], r), 1)
    return df


def resample_count(df: pandas.DataFrame, freq: str, colname: str) -> pandas.DataFrame:
    _df = pandas.DataFrame({colname: df.index}, index=df.index)
    return _df.resample(freq).agg('count')


def groupify(df, freq='1H', nodes=None, min_count=0.1) -> NAryTreeT:
    """

    Parameters
    ----------
    df : pandas.DataFrame
    freq : str
        resample frequency
    nodes : tuple
        Hierarchy node
    min_count : int or float
        Minimum number of observations for a node to be used. If float, it represents a
        fraction of values that need to be non NaNs and must be between 0 and 1
    Returns
    -------

    """
    root_name = 'total'
    total = resample_count(df, freq, root_name)
    hierarchy = HierarchyTree(key=root_name, item=total)

    child_group = nodes[0]                    # city
    children = df[child_group].unique()       # berlin, munich, ...

    # add first level children
    for child in children:
        sub_df = df[df[child_group] == child]
        resampled = resample_count(sub_df, freq, child)
        hierarchy.add_child(key=child, item=resampled, exogenous=None)

    # add the rest
    for node in nodes[1:]:
        parent_group = child_group
        child_group = node                       # hex_index_6
        children = df[child_group].unique()      # abcccc, abccf, ...

        for child in children:
            sub_df = df[df[child_group] == child]
            if isinstance(min_count, float):
                allowance = len(total) * min_count
            elif isinstance(min_count, int):
                allowance = min_count
            else:
                raise InvalidArgumentException(f'min_count must be either float or integer')
            if len(sub_df) < allowance:
                continue
            parent_name = sub_df[parent_group].value_counts().index[0]
            resampled = resample_count(sub_df, freq, child)
            for c in hierarchy.traversal_level():
                if c.key == parent_name:
                    c.add_child(key=child, item=resampled, exogenous=None)
    return hierarchy