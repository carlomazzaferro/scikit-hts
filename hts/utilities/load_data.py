import logging
import os
from io import StringIO

import numpy
import pandas

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:  # pragma: no cover
    logger.error(
        "Some loading functions might be impaired, install requests "
        "with: \npip install requests\n if you'd like to use them"
    )

MOBILITY_URL = "https://hierarchical-sample-data.s3.amazonaws.com/mobility.csv"
GEO_EVENTS_URL = "https://osf.io/v8qax/download"


def get_data_home(data_home=None):
    """
    Return the path of the scikit-hts data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'scikit_hts_data' in the
    user home folder.
    Alternatively, it can be set by the 'SCIKIT_HTS_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    Parameters
    ----------
    data_home : str | None
        The path to scikit-hts data dir.
    """
    if data_home is None:
        data_home = os.environ.get(
            "SCIKIT_HTS_DATA", os.path.join("~", "scikit_hts_data")
        )
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def partition_column(column, n=3):
    partitioned = column.apply(
        lambda x: numpy.random.dirichlet(numpy.ones(n), size=1).ravel() * x
    ).values
    return [[i[j] for i in partitioned] for j in range(n)]


def load_hierarchical_sine_data(start, end, n=10000):
    dts = (end - start).total_seconds()
    dti = pandas.DatetimeIndex(
        [start + pandas.Timedelta(numpy.random.uniform(0, dts), "s") for _ in range(n)]
    ).sort_values()
    time = numpy.arange(0, len(dti), 0.01)
    amplitude = numpy.sin(time) * 10
    amplitude += numpy.random.normal(2 * amplitude + 2, 5)
    df = pandas.DataFrame(index=dti, data={"total": amplitude[0 : len(dti)]})
    df["a"], df["b"], df["c"] = partition_column(df.total, n=3)
    df["a_x"], df["a_y"] = partition_column(df.a, n=2)
    df["b_x"], df["b_y"] = partition_column(df.b, n=2)
    df["c_x"], df["c_y"] = partition_column(df.c, n=2)
    df["a_x_1"], df["a_x_2"] = partition_column(df.a_x, n=2)
    df["a_y_1"], df["a_y_2"] = partition_column(df.a_y, n=2)
    df["b_x_1"], df["b_x_2"] = partition_column(df.b_x, n=2)
    df["b_y_1"], df["b_y_2"] = partition_column(df.b_y, n=2)
    df["c_x_1"], df["c_x_2"] = partition_column(df.c_x, n=2)
    df["c_y_1"], df["c_y_2"] = partition_column(df.c_y, n=2)
    return df


def load_mobility_data(data_home=None):
    """
    Original dataset: https://www.kaggle.com/pronto/cycle-share-dataset
    Returns
    -------
    df : pandas.DataFrame
    """
    data_path = get_data_home(data_home)
    if "mobility.csv" not in os.listdir(data_path):
        df_string = requests.get(MOBILITY_URL).content
        df = pandas.read_csv(
            StringIO(df_string.decode("utf-8")),
            index_col="starttime",
            parse_dates=["starttime"],
        )
        df.reset_index().to_csv(os.path.join(data_path, "mobility.csv"), index=False)
        return df
    else:
        return pandas.read_csv(
            os.path.join(data_path, "mobility.csv"),
            index_col="starttime",
            parse_dates=["starttime"],
        )


def load_geo_events_data(data_home=None):
    """
    Returns
    -------
    df : pandas.DataFrame
    """
    data_path = get_data_home(data_home)
    if "power.csv" not in os.listdir(data_path):
        df_string = requests.get(GEO_EVENTS_URL).content
        df = pandas.read_csv(
            StringIO(df_string.decode("utf-8")),
            parse_dates=["event_ts"],
            index_col="event_ts",
        )
        df.reset_index().to_csv(os.path.join(data_path, "power.csv"), index=False)
        return df
    else:
        return pandas.read_csv(
            os.path.join(data_path, "power.csv"),
            parse_dates=["event_ts"],
            index_col="event_ts",
        )
