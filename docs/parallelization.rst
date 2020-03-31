.. _parallelization-label:

Parallelization
===============

The model fitting as well as the forecasting offer the possibility of parallelization.
Out of the box both tasks are parallelized by scikit-hts. However, the overhead introduced with the
parallelization should not be underestimated. Here we discuss the different settings to control
the parallelization. To achieve best results for your use-case you should experiment with the parameters.


Parallelization of Model Fitting
--------------------------------

We use a :class:`multiprocessing.Pool` to parallelize the fitting of each model to a node's data. On
instantiation we set the Pool's number of worker processes to
`n_jobs`. This field defaults to
the number of processors on the current system. We recommend setting it to the maximum number of available (and
otherwise idle) processors.

The chunksize of the Pool's map function is another important parameter to consider. It can be set via the
`chunksize` field. By default it is up to
:class:`multiprocessing.Pool` is parallelisation parameter. One data chunk is
defined as a singular time series for one node. The chunksize is the
number of chunks that are submitted as one task to one worker process.  If you
set the chunksize to 10, then it means that one worker task corresponds to
calculate all forecasts for 10 node time series.  If it is set it
to None, depending on distributor, heuristics are used to find the optimal
chunksize.  The chunksize can have an crucial influence on the optimal cluster
performance and should be optimised in benchmarks for the problem at hand.

Parallelization of Forecasting
------------------------------

For the feature extraction scikit-hts exposes the parameters
`n_jobs` and `chunksize`. Both behave analogue to the parameters
for the feature selection.

To do performance studies and profiling, it sometimes quite useful to turn off parallelization at all. This can be
setting the parameter `n_jobs` to 0.


Acknowledgement
'''''''''''''''
This documentation, as well as the underlying implementation, exists only thanks to the folks at `blue-yonder`_. The
This page was pretty much copy and pasted from their `tsfresh`_ package. Many thanks for their excellent package.

.. _blue-yonder: https://github.com/blue-yonder
.. _tsfresh: https://github.com/blue-yonder/tsfresh
