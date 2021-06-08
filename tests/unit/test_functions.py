import numpy
import pandas

import hts.hierarchy
from hts.functions import (
    _create_bl_str_col,
    get_agg_series,
    get_hierarchichal_df,
    to_sum_mat,
)


def test_sum_mat_uv(uv_tree):
    mat, sum_mat_labels = to_sum_mat(uv_tree)
    assert isinstance(mat, numpy.ndarray)
    shp = mat.shape
    assert shp[0] == uv_tree.num_nodes() + 1
    assert shp[1] == uv_tree.leaf_sum()


def test_sum_mat_mv(mv_tree):
    mat, _ = to_sum_mat(mv_tree)
    assert isinstance(mat, numpy.ndarray)
    shp = mat.shape
    assert shp[0] == mv_tree.num_nodes() + 1
    assert shp[1] == mv_tree.leaf_sum()


def test_sum_mat_hierarchical():
    hierarchy = {"total": ["A", "B"], "A": ["A_X", "A_Y", "A_Z"], "B": ["B_X", "B_Y"]}
    hier_df = pandas.DataFrame(
        data={
            "total": [],
            "A": [],
            "B": [],
            "A_X": [],
            "A_Y": [],
            "A_Z": [],
            "B_X": [],
            "B_Y": [],
        }
    )

    tree = hts.hierarchy.HierarchyTree.from_nodes(hierarchy, hier_df)
    sum_mat, sum_mat_labels = to_sum_mat(tree)

    expected_sum_mat = numpy.array(
        [
            [1, 1, 1, 1, 1],  # total
            [0, 0, 0, 1, 1],  # B
            [1, 1, 1, 0, 0],  # A
            [1, 0, 0, 0, 0],  # A_X
            [0, 1, 0, 0, 0],  # A_Y
            [0, 0, 1, 0, 0],  # A_Z
            [0, 0, 0, 1, 0],  # B_X
            [0, 0, 0, 0, 1],
        ]
    )  # B_Y

    numpy.testing.assert_array_equal(sum_mat, expected_sum_mat)
    assert sum_mat_labels == ["total", "B", "A", "A_X", "A_Y", "A_Z", "B_X", "B_Y"]


def test_sum_mat_grouped():
    hierarchy = {
        "total": ["A", "B", "X", "Y"],
        "A": ["A_X", "A_Y"],
        "B": ["B_X", "B_Y"],
    }
    grouped_df = pandas.DataFrame(
        data={
            "total": [],
            "A": [],
            "B": [],
            "X": [],
            "Y": [],
            "A_X": [],
            "A_Y": [],
            "B_X": [],
            "B_Y": [],
        }
    )

    tree = hts.hierarchy.HierarchyTree.from_nodes(hierarchy, grouped_df)
    sum_mat, sum_mat_labels = to_sum_mat(tree)

    expected_sum_mat = numpy.array(
        [
            [1, 1, 1, 1],  # total
            [0, 1, 0, 1],  # Y
            [1, 0, 1, 0],  # X
            [0, 0, 1, 1],  # B
            [1, 1, 0, 0],  # A
            [1, 0, 0, 0],  # A_X
            [0, 1, 0, 0],  # A_Y
            [0, 0, 1, 0],  # B_X
            [0, 0, 0, 1],  # B_Y
        ]
    )

    numpy.testing.assert_array_equal(sum_mat, expected_sum_mat)
    assert sum_mat_labels == ["total", "Y", "X", "B", "A", "A_X", "A_Y", "B_X", "B_Y"]


def test_sum_mat_visnights_hier(visnights_hier):
    hier_df = pandas.DataFrame(
        data={
            "total": [],
            "VIC": [],
            "QLD": [],
            "SAU": [],
            "WAU": [],
            "OTH": [],
            "NSW": [],
            "NSW_Metro": [],
            "NSW_NthCo": [],
            "NSW_NthIn": [],
            "NSW_SthCo": [],
            "NSW_SthIn": [],
            "OTH_Metro": [],
            "OTH_NoMet": [],
            "QLD_Cntrl": [],
            "QLD_Metro": [],
            "QLD_NthCo": [],
            "SAU_Coast": [],
            "SAU_Inner": [],
            "SAU_Metro": [],
            "VIC_EstCo": [],
            "VIC_Inner": [],
            "VIC_Metro": [],
            "VIC_WstCo": [],
            "WAU_Coast": [],
            "WAU_Inner": [],
            "WAU_Metro": [],
        }
    )

    tree = hts.hierarchy.HierarchyTree.from_nodes(visnights_hier, hier_df)
    sum_mat, sum_mat_labels = to_sum_mat(tree)

    expected_sum_mat = numpy.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # total
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # VIC
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # QLD
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # SAU
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # WAU
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # OTH
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NSW
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NSW_Metro
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NSW_NthCo
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NSW_NthIn
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NSW_SthCo
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NSW_SthIn
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # OTH_Metro
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # OTH_NoMet
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # WAU_Coast
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # WAU_Inner
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # WAU_Metro
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SAU_Coast
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # SAU_Inner
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # SAU_Metro
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # QLD_Cntrl
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # QLD_Metro
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # QLD_NthCo
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # VIC_EstCo
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # VIC_Inner
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # VIC_Metro
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # VIC_WstCo
        ]
    )

    numpy.testing.assert_array_equal(sum_mat, expected_sum_mat)


def test_demo_unique_constraint():
    # Example https://otexts.com/fpp2/hts.html
    # Does not work when you have elements that are named the same, but represent
    # different levels in the hierarchy. See expected_sum_mat below for example.
    hierarchy = {"total": ["A", "B"], "A": ["AA", "AB", "AC"], "B": ["BA", "BB"]}
    hier_df = pandas.DataFrame(
        data={
            "total": [],
            "A": [],
            "B": [],
            "AA": [],
            "AB": [],
            "AC": [],
            "BA": [],
            "BB": [],
        }
    )

    tree = hts.hierarchy.HierarchyTree.from_nodes(hierarchy, hier_df)
    sum_mat, sum_mat_labels = to_sum_mat(tree)

    expected_sum_mat = numpy.array(
        [
            [1, 1, 1, 1, 1],  # total
            [0, 1, 0, 1, 1],  # B, Incorrectly finds B in AB
            [1, 1, 1, 1, 0],  # A, Incorrectly finds A in BA
            [1, 0, 0, 0, 0],  # AA
            [0, 1, 0, 0, 0],  # AB
            [0, 0, 1, 0, 0],  # AC
            [0, 0, 0, 1, 0],  # BA
            [0, 0, 0, 0, 1],  # BB
        ]
    )

    numpy.testing.assert_array_equal(sum_mat, expected_sum_mat)


def test_1lev():
    grouped_df = pandas.DataFrame(
        data={
            "lev1": ["A", "A", "B", "B"],
            "lev2": ["X", "Y", "X", "Y"],
        }
    )

    levels = get_agg_series(grouped_df, [["lev1"]])
    expected_levels = ["A", "B"]
    assert sorted(levels) == sorted(expected_levels)

    levels = get_agg_series(grouped_df, [["lev2"]])
    expected_levels = ["X", "Y"]
    assert sorted(levels) == sorted(expected_levels)


def test_2lev():
    grouped_df = pandas.DataFrame(
        data={
            "lev1": ["A", "A", "B", "B"],
            "lev2": ["X", "Y", "X", "Y"],
        }
    )

    levels = get_agg_series(grouped_df, [["lev1", "lev2"]])

    expected_levels = ["A_X", "A_Y", "B_X", "B_Y"]

    assert sorted(levels) == sorted(expected_levels)


def test_hierarchichal():
    hier_df = pandas.DataFrame(
        data={
            "lev1": ["A", "A", "A", "B", "B"],
            "lev2": ["X", "Y", "Z", "X", "Y"],
        }
    )

    levels = get_agg_series(hier_df, [["lev1"], ["lev1", "lev2"]])
    expected_levels = ["A", "B", "A_X", "A_Y", "A_Z", "B_X", "B_Y"]
    assert sorted(levels) == sorted(expected_levels)


def test_grouped():
    hier_df = pandas.DataFrame(
        data={
            "lev1": ["A", "A", "A", "B", "B"],
            "lev2": ["X", "Y", "Z", "X", "Y"],
        }
    )

    hierarchy = [["lev1"], ["lev2"], ["lev1", "lev2"]]
    levels = get_agg_series(hier_df, hierarchy)
    expected_levels = ["A", "B", "X", "Y", "Z", "A_X", "A_Y", "A_Z", "B_X", "B_Y"]
    assert sorted(levels) == sorted(expected_levels)


def test_grouped_create_df():
    hier_df = pandas.DataFrame(
        data={
            "ds": ["2020-01", "2020-02"] * 5,
            "lev1": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B"],
            "lev2": ["X", "X", "Y", "Y", "Z", "Z", "X", "X", "Y", "Y"],
            "val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    level_names = ["lev1", "lev2"]
    hierarchy = [["lev1"], ["lev2"]]
    gts_df, sum_mat, sum_mat_labels = get_hierarchichal_df(
        hier_df,
        level_names=level_names,
        hierarchy=hierarchy,
        date_colname="ds",
        val_colname="val",
    )

    expected_columns = [
        "A_X",
        "A_Y",
        "A_Z",
        "B_X",
        "B_Y",
        "A",
        "B",
        "X",
        "Y",
        "Z",
        "total",
    ]
    assert sorted(list(gts_df.columns)) == sorted(expected_columns)


def test_parent_child():
    grouped_df = pandas.DataFrame(
        data={
            "lev1": ["A", "A", "B"],
            "lev2": ["X", "Y", "Z"],
        }
    )

    levels = get_agg_series(grouped_df, [["lev1", "lev2"]])
    expected_levels = ["A_X", "A_Y", "B_Z"]
    assert sorted(levels) == sorted(expected_levels)


def test_create_bl_str_col():
    grouped_df = pandas.DataFrame(
        data={
            "lev1": ["A", "A", "B"],
            "lev2": ["X", "Y", "Z"],
        }
    )

    col = _create_bl_str_col(grouped_df, ["lev1", "lev2"])

    assert col == ["A_X", "A_Y", "B_Z"]
