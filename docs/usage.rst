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

