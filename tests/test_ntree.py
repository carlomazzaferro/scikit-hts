

def test_level_order_traversal(n_tree):
    assert n_tree.level_order_traversal() == [[3], [2, 2, 2], [2, 2, 2, 2, 2, 2]]


def test_get_heights(n_tree):
    assert [sum(i) for i in n_tree.level_order_traversal()] == [3, 6, 12]
