import pandas

from hts.hierarchy import HierarchyTree
from hts._t import NAryTreeT


def test_level_order_traversal(n_tree):
    assert n_tree.level_order_traversal() == [[3], [2, 2, 2], [2, 2, 2, 2, 2, 2]]
    bottom = n_tree.get_node('aab')
    print(bottom)
    for i in bottom.level_order_traversal():
        assert i is None


def test_get_heights(n_tree):
    assert [sum(i) for i in n_tree.level_order_traversal()] == [3, 6, 12]


def test_get_node(n_tree):
    bab = n_tree.get_node('bab')
    assert isinstance(bab, NAryTreeT)
    assert bab.item == 6
    assert bab.is_leaf()
    assert n_tree.is_leaf() is False
    assert n_tree.leaf_sum() == 12
    assert n_tree.get_node('doesnotexist') is None


def test_create_from_events(events):
    ht = HierarchyTree.from_geo_events(df=events,
                                       lat_col='start_latitude',
                                       lon_col='start_longitude',
                                       nodes=('city', 'hex_index_6', 'hex_index_7', 'hex_index_8'),
                                       levels=(6, 8),
                                       resample_freq='1H',
                                       min_count=0.5
                                       )
    assert isinstance(ht, NAryTreeT)
    assert len(ht.children) == events['city'].nunique()


def test_num_nodes(n_tree):
    assert n_tree.num_nodes() == 21


def test_height(n_tree):
    assert n_tree.value_at_height(1) == [3]
    assert n_tree.value_at_height(2) == [2, 2, 2]
    assert n_tree.value_at_height(3) == [2, 2, 2, 2, 2, 2]
    assert n_tree.sum_at_height(3) == 12
    assert n_tree.get_node_height('bb') == 1
    assert n_tree.get_node_height('bbb') == 0
    assert n_tree.get_node_height('a') == 2
    assert n_tree.get_node_height('z') == -1


def test_to_pandas(events):
    ht = HierarchyTree.from_geo_events(df=events,
                                       lat_col='start_latitude',
                                       lon_col='start_longitude',
                                       nodes=('city', 'hex_index_6', 'hex_index_7', 'hex_index_8'),
                                       levels=(6, 8),
                                       resample_freq='1H',
                                       min_count=0.5
                                       )
    assert isinstance(ht.to_pandas(), pandas.DataFrame)
