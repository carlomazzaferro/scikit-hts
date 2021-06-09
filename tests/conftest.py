from datetime import datetime
from io import StringIO

import numpy
import pandas
import pytest

from hts.hierarchy import HierarchyTree
from hts.utilities.load_data import load_hierarchical_sine_data, load_mobility_data


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
    df = pandas.read_csv(StringIO(s), index_col="ts", sep=",")
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

    t = ("t", 1)
    t1 = [("a", 2), ("b", 2), ("c", 3)]
    t2 = [("aa", 4), ("ab", 5), ("ba", 6), ("bb", 4), ("ca", 5), ("cb", 6)]
    t3 = [
        ("aaa", 4),
        ("aab", 5),
        ("aba", 6),
        ("abb", 4),
        ("baa", 5),
        ("bab", 6),
        ("bba", 5),
        ("bbb", 6),
        ("caa", 5),
        ("cab", 6),
        ("cba", 5),
        ("cbb", 6),
    ]

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
    return load_mobility_data()


@pytest.fixture
def mv_tree_empty():
    hier = {
        "total": ["CH", "SLU", "BT", "OTHER"],
        "CH": ["CH-07", "CH-02", "CH-08", "CH-05", "CH-01"],
        "SLU": ["SLU-15", "SLU-01", "SLU-19", "SLU-07", "SLU-02"],
        "BT": ["BT-01", "BT-03"],
        "OTHER": ["WF-01", "CBD-13"],
    }
    return hier


@pytest.fixture
def mv_tree(hierarchical_mv_data, mv_tree_empty):

    exogenous = {
        k: ["precipitation", "temp"]
        for k in hierarchical_mv_data.columns
        if k not in ["precipitation", "temp"]
    }
    return HierarchyTree.from_nodes(
        mv_tree_empty, hierarchical_mv_data, exogenous=exogenous
    )


@pytest.fixture
def sine_hier():
    return {
        "total": ["a", "b", "c"],
        "a": ["a_x", "a_y"],
        "b": ["b_x", "b_y"],
        "c": ["c_x", "c_y"],
        "a_x": ["a_x_1", "a_x_2"],
        "a_y": ["a_y_1", "a_y_2"],
        "b_x": ["b_x_1", "b_x_2"],
        "b_y": ["b_y_1", "b_y_2"],
        "c_x": ["c_x_1", "c_x_2"],
        "c_y": ["c_y_1", "c_y_2"],
    }


@pytest.fixture
def uv_tree(sine_hier, hierarchical_sine_data):
    hsd = hierarchical_sine_data.resample("1H").apply(sum).head(400)
    return HierarchyTree.from_nodes(sine_hier, hsd)


@pytest.fixture
def load_df_and_hier_uv(sine_hier, hierarchical_sine_data):
    return hierarchical_sine_data.resample("1H").apply(sum), sine_hier


@pytest.fixture
def sample_ds():
    cid = numpy.repeat([10, 500], 40)
    ckind = numpy.repeat(["a", "b", "a", "b"], 20)
    csort = [
        30,
        53,
        26,
        35,
        42,
        25,
        17,
        67,
        20,
        68,
        46,
        12,
        0,
        74,
        66,
        31,
        32,
        2,
        55,
        59,
        56,
        60,
        34,
        69,
        47,
        15,
        49,
        8,
        50,
        73,
        23,
        62,
        24,
        33,
        22,
        70,
        3,
        38,
        28,
        75,
        39,
        36,
        64,
        13,
        72,
        52,
        40,
        16,
        58,
        29,
        63,
        79,
        61,
        78,
        1,
        10,
        4,
        6,
        65,
        44,
        54,
        48,
        11,
        14,
        19,
        43,
        76,
        7,
        51,
        9,
        27,
        21,
        5,
        71,
        57,
        77,
        41,
        18,
        45,
        37,
    ]
    cval = [
        11,
        9,
        67,
        45,
        30,
        58,
        62,
        19,
        56,
        29,
        0,
        27,
        36,
        43,
        33,
        2,
        24,
        71,
        41,
        28,
        50,
        40,
        39,
        7,
        53,
        23,
        16,
        37,
        66,
        38,
        6,
        47,
        3,
        61,
        44,
        42,
        78,
        31,
        21,
        55,
        15,
        35,
        25,
        32,
        69,
        65,
        70,
        64,
        51,
        46,
        5,
        77,
        26,
        73,
        76,
        75,
        72,
        74,
        10,
        57,
        4,
        14,
        68,
        22,
        18,
        52,
        54,
        60,
        79,
        12,
        49,
        63,
        8,
        59,
        1,
        13,
        20,
        17,
        48,
        34,
    ]
    df = pandas.DataFrame({"id": cid, "kind": ckind, "sort": csort, "val": cval})
    df = df.set_index("id", drop=False)
    df.index.name = None
    return df


@pytest.fixture
def visnights_hier():
    return {
        "total": ["NSW", "OTH", "WAU", "SAU", "QLD", "VIC"],
        "NSW": ["NSW_Metro", "NSW_NthCo", "NSW_NthIn", "NSW_SthCo", "NSW_SthIn"],
        "OTH": ["OTH_Metro", "OTH_NoMet"],
        "QLD": ["QLD_Cntrl", "QLD_Metro", "QLD_NthCo"],
        "SAU": ["SAU_Coast", "SAU_Inner", "SAU_Metro"],
        "VIC": ["VIC_EstCo", "VIC_Inner", "VIC_Metro", "VIC_WstCo"],
        "WAU": ["WAU_Coast", "WAU_Inner", "WAU_Metro"],
    }


@pytest.fixture
def hierarchical_visnights_data():
    vis_idx = pandas.date_range(start="1998-01-01", periods=8, freq="QS")
    vis_values = {
        "NSW_Metro": [9047, 6962, 6872, 7147, 7957, 6542, 6330, 7509],
        "NSW_NthCo": [8566, 7124, 4717, 6269, 9494, 5401, 5543, 6383],
        "NSW_NthIn": [2978, 3478, 3015, 3758, 3791, 3395, 3626, 3691],
        "NSW_SthCo": [5818, 2466, 1928, 2798, 4854, 2760, 2042, 2651],
        "NSW_SthIn": [2680, 3011, 3329, 2418, 3224, 2428, 2893, 2815],
        "OTH_Metro": [3438, 2677, 3794, 3304, 3511, 2872, 3833, 3143],
        "OTH_NoMet": [2073, 1788, 2345, 1944, 2166, 1804, 1613, 1652],
        "QLD_Cntrl": [2748, 4041, 5344, 4260, 4186, 4238, 6415, 3710],
        "QLD_Metro": [12106, 7787, 11380, 9311, 12672, 9583, 11193, 9871],
        "QLD_NthCo": [2137, 2270, 4890, 2622, 2483, 3378, 5578, 4279],
        "SAU_Coast": [2592, 1376, 1080, 1498, 2248, 1673, 1105, 1503],
        "SAU_Inner": [895, 979, 980, 1509, 964, 997, 1058, 771],
        "SAU_Metro": [2881, 2125, 2285, 1786, 2294, 2197, 2034, 2253],
        "VIC_EstCo": [3382, 1828, 1352, 1493, 2897, 1548, 914, 1342],
        "VIC_Inner": [5327, 4441, 3816, 3860, 4589, 4070, 4114, 3723],
        "VIC_Metro": [7490, 5198, 5244, 6274, 9187, 4992, 4746, 4685],
        "VIC_WstCo": [2442, 961, 756, 1272, 2385, 1329, 759, 942],
        "WAU_Coast": [3067, 3334, 4366, 4522, 3579, 3409, 3979, 3365],
        "WAU_Inner": [695, 558, 1006, 1173, 398, 596, 951, 832],
        "WAU_Metro": [3076, 2155, 2787, 2753, 3520, 3160, 2708, 2294],
        "NSW": [29088, 23041, 19861, 22390, 29320, 20527, 20434, 23049],
        "QLD": [16992, 14097, 21614, 16193, 19341, 17199, 23186, 17861],
        "SAU": [6368, 4480, 4345, 4793, 5505, 4867, 4196, 4526],
        "VIC": [18641, 12428, 11168, 12899, 19058, 11939, 10534, 10692],
        "WAU": [6837, 6047, 8159, 8447, 7497, 7165, 7638, 6491],
        "OTH": [5511, 4465, 6139, 5248, 5677, 4676, 5446, 4796],
        "total": [83437, 64558, 71285, 69971, 86398, 66373, 71434, 67415],
    }

    return pandas.DataFrame(vis_values, index=vis_idx)


@pytest.fixture
def load_df_and_hier_visnights(visnights_hier, hierarchical_visnights_data):
    return hierarchical_visnights_data, visnights_hier
