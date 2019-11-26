from h3 import h3
import pandas


def hexify(df, lat_col, lon_col, levels=(6, 8)):
    for r in range(levels[0], levels[1] + 1):
        df[f'hex_index_{r}'] = df.apply(lambda x: h3.geo_to_h3(x[lat_col], x[lon_col], r), 1)
    return df


def resample_count(df, freq, colname):
    df[colname] = df.index
    _df = df[[colname]]
    return _df.resample(freq).agg('count')


def groupify(df, freq='1H', groups=None):
    total = resample_count(df, freq, 'total')
    children = [total]
    for group in groups:
        unique_in_group = df[group].unique()
        for u in unique_in_group:
            sub_df = df[df[group] == u]
            if len(sub_df) < 1000:
                continue
            resampled = resample_count(sub_df, freq, '_'.join([group, u]))
            children.append(resampled)
    return pandas.concat(children, 1, sort=False)
