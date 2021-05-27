=====
Usage
=====

Typical Usage
-------------

``scikit-hts`` has one main class that provides the interface with your desired forecasting methodology and reconciliation
strategy. Here you can find how to get started quickly with ``scikit-hts``. We'll use some sample (fake) data.


.. code-block:: python

    >>> from datetime import datetime
    >>> from hts import HTSRegressor
    >>> from hts.utilities.load_data import load_hierarchical_sine_data

    # load some data
    >>> s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
    >>> hsd = load_hierarchical_sine_data(s, e).resample('1H').apply(sum)
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

    >>> hsd.head()

                             total         a         b         c         d        aa        ab  ...        ba        bb        bc        ca        cb        cc        cd
    2019-01-15 00:00:00  11.934729  0.638735  3.436469  5.195530  2.663996  0.218140  0.420594  ...  1.449734  1.727512  0.259222  0.593310  1.251554  2.217371  1.133295
    2019-01-15 01:00:00   8.698295  2.005391  2.687024  1.740504  2.265375  0.254958  1.750433  ...  1.963620  0.390856  0.332549  0.566592  0.197838  0.547443  0.428632
    2019-01-15 02:00:00  12.093040  3.802658  2.204833  2.933652  3.151896  3.185786  0.616872  ...  0.110134  1.885216  0.209483  1.332533  0.301493  1.294185  0.005441
    2019-01-15 03:00:00  14.365129  4.332290  3.234713  0.780173  6.017954  3.993601  0.338689  ...  0.846830  0.777724  1.610158  0.091538  0.505417  0.079388  0.103830
    2019-01-15 04:00:00   1.030305  2.073372  0.649284 -1.536231 -0.156119 -0.184177  2.257549  ...  0.433048 -0.179693  0.395928 -0.667796  0.112877 -0.050382 -0.930930


    >>> reg = HTSRegressor(model='prophet', revision_method='OLS')
    >>> reg = reg.fit(df=hsd, nodes=hier)
    >>> preds = reg.predict(steps_ahead=10)


More extensive usage, including a solution for Kaggle's `M5 Competition`_, can be found in the `scikit-hts-examples`_ repo.

.. _M5 Competition: https://www.kaggle.com/c/m5-forecasting-accuracy
.. _scikit-hts-examples: https://github.com/carlomazzaferro/scikit-hts-examples


Ground Up Example
-----------------

Here's a ground up walk through of taking raw data, making custom forecasts, and reconciling them using the example from `FPP <https://otexts.com/fpp3/hts.html>`_.

This small block creates the raw data. We assume a good number of users begin with tabular data coming from database.

.. code-block:: python

    >>> import hts.functions
    >>> import pandas
    >>> import collections

    >>> hier_df = pandas.DataFrame(
        data={
            'ds': ['2020-01', '2020-02'] * 5,
            "lev1": ['A', 'A',
                     'A', 'A',
                     'A', 'A',
                     'B', 'B',
                     'B', 'B'],
            "lev2": ['X', 'X',
                     'Y', 'Y',
                     'Z', 'Z',
                     'X', 'X',
                     'Y', 'Y'],
            "val": [1, 2,
                    3, 4,
                    5, 6,
                    7, 8,
                    9, 10]
            }
        )
    >>> hier_df
            ds lev1 lev2  val
    0  2020-01    A    X    1
    1  2020-02    A    X    2
    2  2020-01    A    Y    3
    3  2020-02    A    Y    4
    4  2020-01    A    Z    5
    5  2020-02    A    Z    6
    6  2020-01    B    X    7
    7  2020-02    B    X    8
    8  2020-01    B    Y    9
    9  2020-02    B    Y   10


Specify a hierarchy of your choosing. Where the ``level_names`` argument is a list of column names that represent levels in the hierarchy.
The ``hierarchy`` argument consists of a list of lists, where you can specify what levels in your hierarchy to include in the hierarchy
structure. You do not need to specify the bottom level of your hierarchy in the ``hierarchy`` argument. This is already included, since
it is equivalent to ``level_names`` aggregation level.

Through the ``hts.function.get_hierarchichal_df`` function you will get a wide ``pandas.DataFrame`` with the individual time series for
you to create forecasts.

.. code-block:: python

    >>> level_names = ['lev1', 'lev2']
    >>> hierarchy = [['lev1'], ['lev2']]
    >>> wide_df, sum_mat, sum_mat_labels = hts.functions.get_hierarchichal_df(hier_df,
                                                                              level_names=level_names,
                                                                              hierarchy=hierarchy,
                                                                              date_colname='ds',
                                                                              val_colname='val')
    >>> wide_df
        lev1_lev2  A_X  A_Y  A_Z  B_X  B_Y  total   A   B   X   Y  Z
        ds
        2020-01      1    3    5    7    9     25   9  16   8  12  5
        2020-02      2    4    6    8   10     30  12  18  10  14  6


Here's an example showing how to easily change your hierarchy, without changing your underlying data.
We do not want to save these results for the sake of following parts of the example.

.. code-block:: python

    >>> hierarchy = [['lev1']]

    >>> a, b, c = hts.functions.get_hierarchichal_df(hier_df,
                                                     level_names=level_names,
                                                     hierarchy=hierarchy,
                                                     date_colname='ds',
                                                     val_colname='val')
    >>> a
    lev1_lev2  A_X  A_Y  A_Z  B_X  B_Y  total   A   B
    ds
    2020-01      1    3    5    7    9     25   9  16
    2020-02      2    4    6    8   10     30  12  18


Create your forecasts and store them in a new DataFrame with the same format. Here we just do an average, but
you can get as complex as you'd like.

.. code-block:: python

    # Create a DataFrame to store new forecasts in
    >>> forecasts = pandas.DataFrame(index=['2020-03'], columns=wide_df.columns)

    >>> import statistics
    >>> for col in wide_df.columns:
            forecasts[col] = statistics.mean(wide_df[col])

    >>> forecasts
    lev1_lev2  A_X  A_Y  A_Z  B_X  B_Y  total     A   B  X   Y    Z
    2020-03    1.5  3.5  5.5  7.5  9.5   27.5  10.5  17  9  13  5.5

Store your forecasts in a dictionary to be passed to the reconciliation algorithm.

.. code-block:: python

    >>> pred_dict = collections.OrderedDict()

    # Add predictions to dictionary is same order as summing matrix
    >>> for label in sum_mat_labels:
        pred_dict[label] = pandas.DataFrame(data=forecasts[label].values, columns=['yhat'])


Reconcile your forecasts. Here we use OLS optimal reconciliation. The, put reconciled forecasts in the same wide DataFrame format.

You'll notice the forecasts are the. Because we used an average to forecast, the forecasts were already coherent. Therefore,
they remain the same/ coherent post-reconciliation. Demonstrating that the reconciliation is working.

.. code-block:: python

    >>> revised = hts.functions.optimal_combination(pred_dict, sum_mat, method='OLS', mse={})

    >>> revised_forecasts = pandas.DataFrame(data=revised[0:,0:],
                                             index=forecasts.index,
                                             columns=sum_mat_labels)

    >>> revised_forecasts
            total     Z     Y    X     B     A  A_X  A_Y  A_Z  B_X  B_Y
    2020-03   27.5  5.5  13.0  9.0  17.0  10.5  1.5  3.5  5.5  7.5  9.5


Reconcile Pre-Computed Forecasts
--------------------------------

This is an example of creating forecasts outside of scikit-hts and then utilzing scikit-hts to do OLS optimal
reconciliation on the forecasts.

.. code-block:: python

    >>> from datetime import datetime
    >>> import hts
    >>> from hts.utilities.load_data import load_hierarchical_sine_data
    >>> import statsmodels
    >>> import collections
    >>> import pandas as pd

    >>> s, e = datetime(2019, 1, 15), datetime(2019, 10, 15)
    >>> hsd = load_hierarchical_sine_data(start=s, end=e, n=10000)
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

    >>> tree = hts.hierarchy.HierarchyTree.from_nodes(hier, hsd)
    >>> sum_mat, sum_mat_labels = hts.functions.to_sum_mat(tree)

    >>> forecasts = pd.DataFrame(columns=hsd.columns, index=['fake'])

        # Make forecasts made outside of package. Could be any modeling technique.
    >>> for col in hsd.columns:
            model = statsmodels.tsa.holtwinters.SimpleExpSmoothing(hsd[col].values).fit()
            fcst = list(model.forecast(1))
            forecasts[col] = fcst

    >>> pred_dict = collections.OrderedDict()

    # Add predictions to dictionary is same order as summing matrix
    >>> for label in sum_mat_labels:
            pred_dict[label] = pd.DataFrame(data=forecasts[label].values, columns=['yhat'])

    >>> revised = hts.functions.optimal_combination(pred_dict, sum_mat, method='OLS', mse={})

    # Put reconciled forecasts in nice DataFrame form
    >>> revised_forecasts = pd.DataFrame(data=revised[0:,0:],
                                        index=forecasts.index,
                                        columns=sum_mat_labels)
