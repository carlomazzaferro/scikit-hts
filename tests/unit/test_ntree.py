import pandas

from hts._t import NAryTreeT
from hts.hierarchy import HierarchyTree


def test_level_order_traversal(n_tree):
    assert n_tree.level_order_traversal() == [[3], [2, 2, 2], [2, 2, 2, 2, 2, 2]]
    bottom = n_tree.get_node("aab")
    for i in bottom.level_order_traversal():
        assert i is None


def test_get_heights(n_tree):
    assert [sum(i) for i in n_tree.level_order_traversal()] == [3, 6, 12]


def test_get_node(n_tree):
    bab = n_tree.get_node("bab")
    assert isinstance(bab, NAryTreeT)
    assert bab.item == 6
    assert bab.is_leaf()
    assert n_tree.is_leaf() is False
    assert n_tree.leaf_sum() == 12
    assert n_tree.get_node("doesnotexist") is None


def test_create_from_events(events):
    ht = HierarchyTree.from_geo_events(
        df=events,
        lat_col="start_latitude",
        lon_col="start_longitude",
        nodes=("city", "hex_index_6", "hex_index_7", "hex_index_8"),
        levels=(6, 8),
        resample_freq="1H",
        min_count=0.5,
    )
    assert isinstance(ht, NAryTreeT)
    assert len(ht.children) == events["city"].nunique()


def test_num_nodes(n_tree):
    assert n_tree.num_nodes() == 21


def test_height(n_tree):
    assert n_tree.value_at_height(1) == [3]
    assert n_tree.value_at_height(2) == [2, 2, 2]
    assert n_tree.value_at_height(3) == [2, 2, 2, 2, 2, 2]
    assert n_tree.sum_at_height(3) == 12
    assert n_tree.get_node_height("bb") == 1
    assert n_tree.get_node_height("bbb") == 0
    assert n_tree.get_node_height("a") == 2
    assert n_tree.get_node_height("z") == -1


def test_to_pandas(events):
    ht = HierarchyTree.from_geo_events(
        df=events,
        lat_col="start_latitude",
        lon_col="start_longitude",
        nodes=("city", "hex_index_6", "hex_index_7", "hex_index_8"),
        levels=(6, 8),
        resample_freq="1H",
        min_count=0.5,
    )
    assert isinstance(ht.to_pandas(), pandas.DataFrame)


def test_from_geo_events(events):
    ht = HierarchyTree.from_geo_events(
        df=events,
        lat_col="start_latitude",
        lon_col="start_longitude",
        nodes=("city", "hex_index_6", "hex_index_7", "hex_index_8"),
        levels=(6, 8),
        resample_freq="1H",
        min_count=0.5,
        fillna=True,
    )
    assert isinstance(ht.to_pandas(), pandas.DataFrame)


def test_create_hierarchical_sine_data_tree(hierarchical_sine_data):
    hier = {
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
    ht = HierarchyTree.from_nodes(hier, hierarchical_sine_data)
    assert isinstance(ht.to_pandas(), pandas.DataFrame)
    assert ht.key == "total"
    assert len(ht.children) == 3
    for c in ht.children:
        if c.key == "a" or c.key == "b" or c.key == "c":
            assert len(c.children) == 2
        if (
            c.key == "a_x"
            or c.key == "b_x"
            or c.key == "c_x"
            or c.key == "a_y"
            or c.key == "b_y"
            or c.key == "c_y"
        ):
            assert len(c.children) == 4


def test_create_mv_tree(hierarchical_mv_data):

    hier = {
        "total": ["CH", "SLU", "BT", "OTHER"],
        "CH": ["CH-07", "CH-02", "CH-08", "CH-05", "CH-01"],
        "SLU": ["SLU-15", "SLU-01", "SLU-19", "SLU-07", "SLU-02"],
        "BT": ["BT-01", "BT-03"],
        "OTHER": ["WF-01", "CBD-13"],
    }
    exogenous = {
        k: ["precipitation", "temp"]
        for k in hierarchical_mv_data.columns
        if k not in ["precipitation", "temp"]
    }

    ht = HierarchyTree.from_nodes(hier, hierarchical_mv_data, exogenous=exogenous)
    assert isinstance(ht.to_pandas(), pandas.DataFrame)
    assert ht.key == "total"
    assert len(ht.children) == 4
    assert ht.get_node_height("CH") == 1
    assert ht.get_node_height("BT-03") == 0
    assert ht.get_node_height("CBD-13") == 0
    assert ht.get_node_height("SLU") == 1
