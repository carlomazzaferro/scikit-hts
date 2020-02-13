import logging
import os
from io import StringIO

import numpy
import pandas

logger = logging.getLogger(__name__)

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


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
