from io import StringIO

import numpy
import pandas

from hts import logger
try:
    import requests
except ImportError:
    logger.error('Some loading functions might be impaired, install requrests '
                 'with: \npip install requests\n if you\'d like to use them')


def partition_column(column, n=3):
    partitioned = column.apply(lambda x: numpy.random.dirichlet(numpy.ones(n),size=1).ravel() * x).values
    return [[i[j] for i in partitioned] for j in range(n)]


def load_hierarchical_sine_data(start, end, n=10000):
    dts = (end - start).total_seconds()
    dti = pandas.DatetimeIndex([start + pandas.Timedelta(numpy.random.uniform(0, dts), 's')
                                for _ in range(n)]).sort_values()
    time = numpy.arange(0, len(dti), 0.01)
    amplitude = numpy.sin(time) * 10
    amplitude += numpy.random.normal(2 * amplitude + 2, 5)
    df = pandas.DataFrame(index=dti, data={'total': amplitude[0:len(dti)]})
    df['a'], df['b'], df['c'], df['d'] = partition_column(df.total, n=4)
    df['aa'], df['ab'] = partition_column(df.a, n=2)
    df['aaa'], df['aab'] = partition_column(df.aa, n=2)
    df['ba'], df['bb'], df['bc'] = partition_column(df.b, n=3)
    df['ca'], df['cb'], df['cc'], df['cd'] = partition_column(df.c, n=4)
    return df


def load_sample_hierarchical_mv_data():
    """
    Original dataset: https://www.kaggle.com/pronto/cycle-share-dataset
    Returns
    -------
    df : pandas.DataFrame
    """
    df = requests.get('https://hierarchical-sample-data.s3.amazonaws.com/mobility.csv').content
    return pandas.read_csv(StringIO(df.decode('utf-8')), index_col='starttime', parse_dates=True)