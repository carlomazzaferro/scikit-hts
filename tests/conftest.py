import pytest

from hts.helpers.hierarchy import HierarchyTree


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
