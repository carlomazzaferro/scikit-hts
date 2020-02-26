Hierarchical Representation
===========================

``scikit-hts``'s core data structure is the ``HierarchyTree``. At its core, it is simply an `N-Ary Tree`_, a recursive
data structure where each node is specified by:

- A human readable key, such as 'germany', 'total', 'berlin', or '881f15ad61fffff'
- An item, represented by a ``pandas.Series`` (or ``pandas.DataFrame`` for multivariate inputs), which contains the
  actual data about that node


.. _`N-Ary Tree`: https://en.wikipedia.org/wiki/M-ary_tree


For instance, a tree with nodes:

.. code-block:: python

    nodes = {'t': ['a', 'b', 'c'],
             'a': ['aa', 'ab'],
             'b': ['ba', 'bb'],
             'c': ['ca', 'cb'],
             'aa': ['aaa', 'aab'],
             'ab': ['aba', 'abb']
             ...
             'cb': ['cba', 'cbb']
             }


Represents the following structure:

.. code-block:: console

    Level                             Node Key                          # of nodes

      1                                  t                                   1

      2                 a                b                   c               3

      3           aa   ab             ba   bb             ca   cb            6

      4      aaa aab   aba abb   baa bab   bba bbb   caa cab   cba cbb       12



To get a sense of how the hierarchy trees are implemented, some sample data can be loaded:

.. code-block:: python

    >>> from datetime import datetime
    >>> from hts import HierarchyTree
    >>> from hts.utils import load_hierarchical_sine_data

    >>> s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
    >>> hsd = load_hierarchical_sine_data(start=s, end=e, n=10000)
    >>> print(hsd.head())
                                    total         a         b         c         d        aa        ab       aaa       aab        ba        bb        bc        ca        cb        cc        cd
    2019-01-15 01:29:25.005972   6.345796  1.500952  2.006216  0.016688  2.821940  1.413739  0.087213  0.273000  1.140739  0.572872  0.438739  0.994606  0.008490  0.003722  0.004431  0.000045
    2019-01-15 01:45:50.195453   9.107371  1.116805  1.091745  5.688870  1.209951  0.291894  0.824912  0.149041  0.142853  0.007558  0.374915  0.709272  1.303977  0.775971  0.288751  3.320171
    2019-01-15 02:20:51.204587  -6.333233 -1.081240 -0.455464 -2.401480 -2.395049 -0.716773 -0.364467 -0.243496 -0.473276 -0.136318 -0.159603 -0.159543 -0.417023 -0.117741 -1.773234 -0.093482
    2019-01-15 02:27:46.966530  -2.432930 -0.348840 -0.207461 -0.851828 -1.024801 -0.317890 -0.030949 -0.175013 -0.142877 -0.034511 -0.006034 -0.166916 -0.286929 -0.329183 -0.005672 -0.230045
    2019-01-15 02:32:09.675895  10.925181  3.820450  1.349626  1.002597  4.752509  3.355709  0.464741  1.596091  1.759618  0.125829  1.206414  0.017383  0.112833  0.515650  0.077102  0.297012

    >>> hier = {'total': ['a', 'b', 'c'], 'a': ['aa', 'ab'], 'aa': ['aaa', 'aab'], 'b': ['ba', 'bb'], 'c': ['ca', 'cb', 'cc', 'cd']}
    >>> tree = HierarchyTree.create_node(hier, hsd, root='total')
    >>> print(tree)
    - total
       |- a
       |  |- aa
       |  |  |- aaa
       |  |  - aab
       |  - ab
       |- b
       |  |- ba
       |  - bb
       - c
          |- ca
          |- cb
          |- cc
          - cd


.. automodule:: hts.hierarchy
    :members:

