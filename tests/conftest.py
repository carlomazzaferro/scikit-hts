from datetime import datetime
from io import StringIO

import numpy
import pandas
import pytest

from hts.hierarchy import HierarchyTree
from hts.utilities.load_data import load_hierarchical_sine_data, load_sample_hierarchical_mv_data


@pytest.fixture
def events():
    s = """ts,start_latitude,start_longitude,city
    2019-12-06 12:29:16.789,53.565173,9.959418,hamburg
    2019-12-06 12:28:37.326,50.120962,8.674268,frankfurt
    2019-12-06 12:27:07.055,52.521168,13.410618,berlin
    2019-12-06 12:26:25.989,51.492683,7.417612,dortmund
    2019-12-06 12:25:40.222,52.537730,13.417372,berlin
    2019-12-06 12:25:25.309,50.948847,6.951802,cologne
    2019-12-06 12:23:53.633,48.166799,11.577420,munich
    2019-12-06 12:23:05.292,50.113883,8.675192,frankfurt
    2019-12-06 12:22:56.059,50.114847,8.672653,frankfurt
    2019-12-06 12:22:39.471,50.943082,6.959962,cologne"""
    df = pandas.read_csv(StringIO(s), index_col='ts', sep=',')
    df.index = pandas.to_datetime(df.index)
    return df


@pytest.fixture
def n_tree():
    """
    This is the format of this tree
                                t                                1

               a                b                   c            3

         aa   ab             ba   bb             ca   cb         6

    aaa aab   aba abb   baa bab   bba bbb   caa cab   cba cbb    12

    Resulting in the summing matrix: y_t = S * b_t

    t        1 1 1 1 1 1 1 1 1 1 1 1
    a        1 1 1 1 0 0 0 0 0 0 0 0
    b        0 0 0 0 1 1 1 1 0 0 0 0
    c        0 0 0 0 0 0 0 0 1 1 1 1
    aa       1 1 0 0 0 0 0 0 0 0 0 0
    ab       0 0 1 1 0 0 0 0 0 0 0 0      aaa
    ba       0 0 0 0 1 1 0 0 0 0 0 0      aab
    bb       0 0 0 0 0 0 1 1 0 0 0 0      aba
    ca       0 0 0 0 0 0 0 0 1 1 0 0      abb
    cb       0 0 0 0 0 0 0 0 0 0 1 1      baa
    aaa      1 0 0 0 0 0 0 0 0 0 0 0      bab
    aab      0 1 0 0 0 0 0 0 0 0 0 0      bba
    aba      0 0 1 0 0 0 0 0 0 0 0 0      bbb
    abb      0 0 0 1 0 0 0 0 0 0 0 0      caa
    baa      0 0 0 0 1 0 0 0 0 0 0 0      cab
    bab      0 0 0 0 0 1 0 0 0 0 0 0      cba
    bba      0 0 0 0 0 0 1 0 0 0 0 0      cbb
    bbb      0 0 0 0 0 0 0 1 0 0 0 0
    caa      0 0 0 0 0 0 0 0 1 0 0 0
    cab      0 0 0 0 0 0 0 0 0 1 0 0
    cba      0 0 0 0 0 0 0 0 0 0 1 0
    cbb      0 0 0 0 0 0 0 0 0 0 0 1

    """

    t = ('t', 1)
    t1 = [('a', 2), ('b', 2), ('c', 3)]
    t2 = [('aa', 4), ('ab', 5), ('ba', 6), ('bb', 4), ('ca', 5), ('cb', 6)]
    t3 = [('aaa', 4), ('aab', 5), ('aba', 6), ('abb', 4), ('baa', 5),
          ('bab', 6), ('bba', 5), ('bbb', 6), ('caa', 5), ('cab', 6),
          ('cba', 5), ('cbb', 6)]

    test_t = HierarchyTree(key=t[0], item=t[1])
    for i, j in t1:
        test_t.add_child(key=i, item=j)

    for c in test_t.children:
        for i, j in t2:
            if i.startswith(c.key):
                c.add_child(key=i, item=j)

    for c in test_t.children:
        for c2 in c.children:
            for i, j in t3:
                if i.startswith(c2.key):
                    c2.add_child(key=i, item=j)
    return test_t


@pytest.fixture
def hierarchical_sine_data():
    s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
    return load_hierarchical_sine_data(s, e)


@pytest.fixture
def hierarchical_mv_data():
    return load_sample_hierarchical_mv_data()


@pytest.fixture
def mv_tree(hierarchical_mv_data):
    hier = {
        'total': ['CH', 'SLU', 'BT', 'OTHER'],
        'CH': ['CH-07', 'CH-02', 'CH-08', 'CH-05', 'CH-01'],
        'SLU': ['SLU-15', 'SLU-01', 'SLU-19', 'SLU-07', 'SLU-02'],
        'BT': ['BT-01', 'BT-03'],
        'OTHER': ['WF-01', 'CBD-13']
    }
    exogenous = {k: ['precipitation', 'temp'] for k in hierarchical_mv_data.columns if
                 k not in ['precipitation', 'temp']}
    return HierarchyTree.from_nodes(hier, hierarchical_mv_data, exogenous=exogenous)


@pytest.fixture
def sine_hier():
    return {'total': ['a', 'b', 'c'],
            'a': ['aa', 'ab'], 'aa': ['aaa', 'aab'],
            'b': ['ba', 'bb'],
            'c': ['ca', 'cb', 'cc', 'cd']}


@pytest.fixture
def uv_tree(sine_hier, hierarchical_sine_data):
    hsd = hierarchical_sine_data.resample('1H').apply(sum).head(400)
    return HierarchyTree.from_nodes(sine_hier, hsd)


@pytest.fixture
def load_df_and_hier_uv(sine_hier, hierarchical_sine_data):
    return hierarchical_sine_data.resample('1H').apply(sum), sine_hier


@pytest.fixture
def sample_ds():
    cid = numpy.repeat([10, 500], 40)
    ckind = numpy.repeat(["a", "b", "a", "b"], 20)
    csort = [30, 53, 26, 35, 42, 25, 17, 67, 20, 68, 46, 12, 0, 74, 66, 31, 32,
             2, 55, 59, 56, 60, 34, 69, 47, 15, 49, 8, 50, 73, 23, 62, 24, 33,
             22, 70, 3, 38, 28, 75, 39, 36, 64, 13, 72, 52, 40, 16, 58, 29, 63,
             79, 61, 78, 1, 10, 4, 6, 65, 44, 54, 48, 11, 14, 19, 43, 76, 7,
             51, 9, 27, 21, 5, 71, 57, 77, 41, 18, 45, 37]
    cval = [11, 9, 67, 45, 30, 58, 62, 19, 56, 29, 0, 27, 36, 43, 33, 2, 24,
            71, 41, 28, 50, 40, 39, 7, 53, 23, 16, 37, 66, 38, 6, 47, 3, 61,
            44, 42, 78, 31, 21, 55, 15, 35, 25, 32, 69, 65, 70, 64, 51, 46, 5,
            77, 26, 73, 76, 75, 72, 74, 10, 57, 4, 14, 68, 22, 18, 52, 54, 60,
            79, 12, 49, 63, 8, 59, 1, 13, 20, 17, 48, 34]
    df = pandas.DataFrame({"id": cid, "kind": ckind, "sort": csort, "val": cval})
    df = df.set_index("id", drop=False)
    df.index.name = None
    return df
