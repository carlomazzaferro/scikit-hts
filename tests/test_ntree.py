from hts.algos.foreacast.helpers import level_order_traversal


def test_level_order_traversal(n_tree):
    assert level_order_traversal(n_tree) == [
        ['a', 'b', 'c'],
        ['aa', 'ab', 'ba', 'bb', 'ca', 'cb'],
        ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba', 'bbb', 'caa', 'cab', 'cba', 'cbb']]


def test_get_heights(n_tree):
    assert [n_tree.sum_at_height(i)
            for i in reversed(range(1, n_tree.get_height()))] == [3, 6, 12]
