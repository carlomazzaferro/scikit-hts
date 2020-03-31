.. role:: python(code)
    :language: python

How to deploy scikit-hts at scale
=================================

The high volume of time series data can demand an analysis at scale.
So, time series need to be processed on a group of computational units instead of a singular machine.

Accordingly, it may be necessary to distribute the extraction of time series features to a cluster.
Indeed, it is possible to extract features with *hts* in a distributed fashion.
This page will explain how to setup a distributed *hts*.

The distributor class
'''''''''''''''''''''

To distribute the calculation of features, we use a certain object, the Distributor class (contained in the
:mod:`hts.utilities.distribution` module).

Essentially, a Distributor organizes the application of feature calculators to data chunks.
It maps the feature calculators to the data chunks and then reduces them, meaning that it combines the results of the
individual mapping into one object, the feature matrix.

So, Distributor will, in the following order,

    1. calculates an optimal :python:`chunk_size`, based on the characteristics of the time series data at hand
       (by :func:`~hts.utilities.distribution.DistributorBaseClass.calculate_best_chunk_size`)

    2. split the time series data into chunks
       (by :func:`~hts.utilities.distribution.DistributorBaseClass.partition`)

    3. distribute the applying of the feature calculators to the data chunks
       (by :func:`~hts.utilities.distribution.DistributorBaseClass.distribute`)

    4. combine the results into the feature matrix
       (by :func:`~hts.utilities.distribution.DistributorBaseClass.map_reduce`)

    5. close all connections, shutdown all resources and clean everything
       (by :func:`~hts.utilities.distribution.DistributorBaseClass.close`)

So, how can you use such a Distributor to extract features with *hts*?
You will have to pass it into as the :python:`distributor` argument to the :func:`~hts.feature_extraction.extract_features`
method.


The following example shows how to define the MultiprocessingDistributor, which will distribute the calculations to a
local pool of threads:

.. code:: python

    from hts import HTSRegressor
    from hts.utilities.load_data import load_mobility_data
    from hts.utilities.distribution import MultiprocessingDistributor

    df = load_mobility_data()

    # Define hierarchy
    hier = {
        'total': ['CH', 'SLU', 'BT', 'OTHER'],
        'CH': ['CH-07', 'CH-02', 'CH-08', 'CH-05', 'CH-01'],
        'SLU': ['SLU-15', 'SLU-01', 'SLU-19', 'SLU-07', 'SLU-02'],
        'BT': ['BT-01', 'BT-03'],
        'OTHER': ['WF-01', 'CBD-13']
    }

    distributor = MultiprocessingDistributor(n_workers=4,
                                             disable_progressbar=False,
                                             progressbar_title="Feature Extraction")
    hts.fit(df=df, nodes=hier, n_jobs=4, distributor=distributor)

This example actually corresponds to the existing multiprocessing API, where you just specify the number of
jobs, without the need to construct the Distributor:

.. code:: python

    from hts import HTSRegressor
    from hts.utilities.load_data import load_mobility_data

    df = load_mobility_data()

    # Define hierarchy
    hier = {
        'total': ['CH', 'SLU', 'BT', 'OTHER'],
        'CH': ['CH-07', 'CH-02', 'CH-08', 'CH-05', 'CH-01'],
        'SLU': ['SLU-15', 'SLU-01', 'SLU-19', 'SLU-07', 'SLU-02'],
        'BT': ['BT-01', 'BT-03'],
        'OTHER': ['WF-01', 'CBD-13']
    }

    hts.fit(df=df, nodes=hier, n_jobs=4)


Using dask to distribute the calculations
'''''''''''''''''''''''''''''''''''''''''

We provide distributor for the `dask framework <https://dask.pydata.org/en/latest/>`_, where
*"Dask is a flexible parallel computing library for analytic computing."*

Dask is a great framework to distribute analytic calculations to a cluster.
It scales up and down, meaning that you can even use it on a singular machine.
The only thing that you will need to run *hts* on a Dask cluster is the ip address and port number of the
`dask-scheduler <http://distributed.readthedocs.io/en/latest/setup.html>`_.

Lets say that your dask scheduler is running at ``192.168.0.1:8786``, then we can easily construct a
:class:`~hts.utilities.distribution.ClusterDaskDistributor` that connects to the scheduler and distributes the
time series data and the calculation to a cluster:

.. code:: python

    from hts import HTSRegressor
    from hts.utilities.load_data import load_mobility_data
    from hts.utilities.distribution import ClusterDaskDistributor


    df = load_mobility_data()

    # Define hierarchy
    hier = {
        'total': ['CH', 'SLU', 'BT', 'OTHER'],
        'CH': ['CH-07', 'CH-02', 'CH-08', 'CH-05', 'CH-01'],
        'SLU': ['SLU-15', 'SLU-01', 'SLU-19', 'SLU-07', 'SLU-02'],
        'BT': ['BT-01', 'BT-03'],
        'OTHER': ['WF-01', 'CBD-13']
    }

    distributor = ClusterDaskDistributor(address="192.168.0.1:8786")
    hts.fit(df=df, nodes=hier)
    ...

    # Prediction also runs in a distributed fashion
    preds = hts.predict(steps_ahead=10)


Compared to the :class:`~hts.utilities.distribution.MultiprocessingDistributor` example from above, we only had to
change one line to switch from one machine to a whole cluster.
It is as easy as that.
By changing the Distributor you can easily deploy your application to run to a cluster instead of your workstation.

You can also use a local DaskCluster on your local machine to emulate a Dask network.
The following example shows how to setup a :class:`~hts.utilities.distribution.LocalDaskDistributor` on a local cluster
of 3 workers:

.. code:: python

    from hts import HTSRegressor
    from hts.utilities.load_data import load_mobility_data
    from hts.utilities.distribution import LocalDaskDistributor


    df = load_mobility_data()

    # Define hierarchy
    hier = {
        'total': ['CH', 'SLU', 'BT', 'OTHER'],
        'CH': ['CH-07', 'CH-02', 'CH-08', 'CH-05', 'CH-01'],
        'SLU': ['SLU-15', 'SLU-01', 'SLU-19', 'SLU-07', 'SLU-02'],
        'BT': ['BT-01', 'BT-03'],
        'OTHER': ['WF-01', 'CBD-13']
    }

    distributor = LocalDaskDistributor(n_workers=3)
    hts.fit(df=df, nodes=hier)
    ...

    # Prediction also runs in a distributed fashion
    preds = hts.predict(steps_ahead=10)


Writing your own distributor
''''''''''''''''''''''''''''

If you want to user another framework than Dask, you will have to write your own Distributor.
To construct your custom Distributor, you will have to define an object that inherits from the abstract base class
:class:`hts.utilities.distribution.DistributorBaseClass`.
The :mod:`hts.utilities.distribution` module contains more information about what you will need to implement.


Acknowledgement
'''''''''''''''
This documentation, as well as the underlying implementation, exists only thanks to the folks at `blue-yonder`_. The
This page was pretty much copy and pasted from their `tsfresh`_ package. Many thanks for their excellent package.

.. _blue-yonder: https://github.com/blue-yonder
.. _tsfresh: https://github.com/blue-yonder/tsfresh




