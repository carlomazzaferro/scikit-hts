from h3 import h3
import pandas

from hts.helpers.hierarchy import NTree


def hexify(df, lat_col, lon_col, levels=(6, 8)):
    for r in range(levels[0], levels[1] + 1):
        df[f'hex_index_{r}'] = df.apply(lambda x: h3.geo_to_h3(x[lat_col], x[lon_col], r), 1)
    return df


def resample_count(df, freq, colname):
    df[colname] = df.index
    _df = df[[colname]]
    return _df.resample(freq).agg('count')


def groupify(df, freq='1H', nodes=None, min_count=None, root_name='total'):
    """

    Parameters
    ----------
    df : pandas.DataFrame
    freq : str
        resample frequency
    nodes : tuple
        Hierarchy node
    min_count : int
        Minimum number of observations for a node to be used
    root_name : str
        Name of the root node. Defaults to `total`

    Returns
    -------

    """

    total = resample_count(df, freq, root_name)
    hierarchy = NTree(key=root_name, item=total)

    child_group = nodes[0]                    # city
    children = df[child_group].unique()       # berlin, munich, ...

    # add first level children
    for child in children:
        sub_df = df[df[child_group] == child]
        resampled = resample_count(sub_df, freq, child)
        for c in hierarchy.traversal(visit=lambda x: x):  # [root_name]
            c.add_child(key=child, item=resampled)

    # add the rest
    for node in nodes[1:]:
        parent_group = child_group
        child_group = node                       # hex_index_6
        children = df[child_group].unique()      # abcccc, abccf, ...

        for child in children:
            sub_df = df[df[child_group] == child]
            if min_count:
                if len(sub_df) < min_count:
                    continue
            parent_name = sub_df[parent_group].value_counts().index[0]
            resampled = resample_count(sub_df, freq, child)
            for c in hierarchy.traversal(visit=lambda x: x):
                if c.key == parent_name:
                    c.add_child(key=child, item=resampled)

    return hierarchy


def hierarchy_to_pandas(hierarchy: NTree) -> pandas.DataFrame:
    return pandas.concat([c.item for c in hierarchy.traversal(lambda x: x)])
