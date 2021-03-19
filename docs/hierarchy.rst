Hierarchical Representation
===========================

``scikit-hts``'s core data structure is the ``HierarchyTree``. At its core, it is simply an `N-Ary Tree`_, a recursive
data structure where each node is specified by:

- A human readable key, such as 'germany', 'total', 'berlin', or '881f15ad61fffff'
- Keys should be unique and delimited by underscores. Therfore, using the example below there should not be duplicate values across level 1, 2 or 3.
  For example, ``a`` should not also a value in level 2.
- An item, represented by a ``pandas.Series`` (or ``pandas.DataFrame`` for multivariate inputs), which contains the
  actual data about that node


.. _`N-Ary Tree`: https://en.wikipedia.org/wiki/M-ary_tree

Hierarchical Structure
----------------------

For instance, a tree with nodes and levels as follows:

- Level 1: a, b, c
- Level 2: x, y
- Level 3: 1, 2


.. code-block:: python

    nodes = {'total': ['a', 'b', 'c'],
             'a': ['a_x', 'a_y'],
             'b': ['b_x', 'b_y'],
             'c': ['c_x', 'c_y'],
             'a_x': ['a_x_1', 'a_x_2'],
             'a_y': ['a_y_1', 'a_y_2'],
             'b_x': ['b_x_1', 'b_x_2'],
             'b_y': ['b_y_1', 'b_y_2'],
             'c_x': ['c_x_1', 'c_x_2'],
             'c_y': ['c_y_1', 'c_y_2']
             }


Represents the following structure:

.. code-block:: console

    Level                                           Node Key                                             # of nodes

      1                                                t                                                      1

      2                    a                           b                             c                        3

      3                a_x   a_y                    b_x   b_y                    c_x   c_y                    6

      4        a_x_1 a_x_2   a_y_1 a_y_2    b_x_1 b_x_2   b_y_1 b_y_2    c_x_1 c_x_2   c_y_1 c_y_2            12



To get a sense of how the hierarchy trees are implemented, some sample data can be loaded:

.. code-block:: python

    >>> from datetime import datetime
    >>> from hts.hierarchy import HierarchyTree
    >>> from hts.utilities.load_data import load_hierarchical_sine_data

    >>> s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
    >>> hsd = load_hierarchical_sine_data(start=s, end=e, n=10000)
    >>> print(hsd.head())
                                    total         a         b         c       a_x       a_y       b_x       b_y       c_x  ...     a_y_2     b_x_1     b_x_2     b_y_1     b_y_2     c_x_1     c_x_2     c_y_1     c_y_2
    2019-01-15 01:11:09.255573   2.695133  0.150805  0.031629  2.512698  0.037016  0.113789  0.028399  0.003231  0.268406  ...  0.080803  0.013131  0.015268  0.000952  0.002279  0.175671  0.092734  0.282259  1.962034
    2019-01-15 01:18:30.753096  -3.274595 -0.199276 -1.624369 -1.450950 -0.117717 -0.081559 -0.300076 -1.324294 -1.340172  ... -0.077289 -0.177000 -0.123075 -0.178258 -1.146035 -0.266198 -1.073975 -0.083517 -0.027260
    2019-01-15 01:57:48.607109  -1.898038 -0.226974 -0.662317 -1.008747 -0.221508 -0.005466 -0.587826 -0.074492 -0.929464  ... -0.003297 -0.218128 -0.369698 -0.021156 -0.053335 -0.225994 -0.703470 -0.077021 -0.002262
    2019-01-15 02:06:57.994575  13.904908  6.025506  5.414178  2.465225  5.012228  1.013278  4.189432  1.224746  1.546544  ...  0.467630  1.297829  2.891602  0.671085  0.553661  0.066278  1.480266  0.769954  0.148728
    2019-01-15 02:14:22.367818  11.028013  3.537919  6.504104  0.985990  2.935614  0.602305  4.503611  2.000493  0.179114  ...  0.091993  4.350293  0.153318  1.349629  0.650864  0.066946  0.112168  0.473987  0.332889


    >>> hier = {'total': ['a', 'b', 'c'],
                'a': ['a_x', 'a_y'],
                'b': ['b_x', 'b_y'],
                'c': ['c_x', 'c_y'],
                'a_x': ['a_x_1', 'a_x_2'],
                'a_y': ['a_y_1', 'a_y_2'],
                'b_x': ['b_x_1', 'b_x_2'],
                'b_y': ['b_y_1', 'b_y_2'],
                'c_x': ['c_x_1', 'c_x_2'],
                'c_y': ['c_y_1', 'c_y_2']
            }
    >>> tree = HierarchyTree.from_nodes(hier, hsd, root='total')
    >>> print(tree)
    - total
       |- a
       |  |- a_x
       |  |  |- a_x_1
       |  |  - a_x_2
       |  - a_y
       |     |- a_y_1
       |     - a_y_2
       |- b
       |  |- b_x
       |  |  |- b_x_1
       |  |  - b_x_2
       |  - b_y
       |    |- b_y_1
       |    - b_y_2
       - c
         |- c_x
         |  |- c_x_1
         |  - c_x_2
         - c_y
            |- c_y_1
            - c_y_2


Grouped Structure
-----------------

In order to create a grouped structure, instead of a strictly hierarchichal structure you must specify
all levels within the grouping strucure dictionary and dataframe as seen below.

Levels in example:

- Level 1: A, B
- Level 2: X, Y

.. code-block:: python

    import hts
    import pandas as pd

    >>> hierarchy = {
        "total": ["A", "B", "X", "Y"],
        "A": ["A_X", "A_Y"],
        "B": ["B_X", "B_Y"],
    }

    >>> grouped_df = pd.DataFrame(
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

    >>> tree = hts.hierarchy.HierarchyTree.from_nodes(hierarchy, grouped_df)
    >>> sum_mat, sum_mat_labels = hts.functions.to_sum_mat(tree)
    >>> print(sum_mat)  # Commented labels will not appear in the printout, they are here as an example.
    [[1. 1. 1. 1.]  # totals
     [0. 1. 0. 1.]  # Y
     [1. 0. 1. 0.]  # X
     [0. 0. 1. 1.]  # B
     [1. 1. 0. 0.]  # A
     [1. 0. 0. 0.]  # A_X
     [0. 1. 0. 0.]  # A_Y
     [0. 0. 1. 0.]  # B_X
     [0. 0. 0. 1.]] # B_Y

     >>> print(sum_mat_labels)  # Use this if you need to match summing matrix rows with labels.
     ['total', 'Y', 'X', 'B', 'A', 'A_X', 'A_Y', 'B_X', 'B_Y']


.. automodule:: hts.hierarchy
    :members:

