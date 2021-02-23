import numpy
import pandas

import hts.hierarchy
from hts.functions import to_sum_mat


def test_sum_mat_uv(uv_tree):
    mat, sum_mat_labels = to_sum_mat(uv_tree)
    assert isinstance(mat, numpy.ndarray)
    shp = mat.shape
    assert shp[0] == uv_tree.num_nodes() + 1
    assert shp[1] == uv_tree.leaf_sum()


def test_sum_mat_mv(mv_tree):
    mat, sum_mat_labels = to_sum_mat(mv_tree)
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
