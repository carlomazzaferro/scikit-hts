import numpy

from hts.functions import to_sum_mat


def test_sum_mat_uv(uv_tree):
    mat = to_sum_mat(uv_tree)
    assert isinstance(mat, numpy.ndarray)
    shp = mat.shape
    assert shp[0] == uv_tree.num_nodes() + 1
    assert shp[1] == uv_tree.leaf_sum()


def test_sum_mat_mv(mv_tree):
    mat = to_sum_mat(mv_tree)
    assert isinstance(mat, numpy.ndarray)
    shp = mat.shape
    assert shp[0] == mv_tree.num_nodes() + 1
    assert shp[1] == mv_tree.leaf_sum()