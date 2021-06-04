##########
scikit-hts
##########

Hierarchical Time Series with a familiar API. This is the result from not having found any good implementations
of HTS on-line, and my work in the mobility space while working at Circ (acquired by Bird scooters).

My work on this is purely out of passion, so contributions are always
welcomed. You can also buy me a coffee if you'd like:


     ETH / BSC Address:  ``0xbF42b9c8F7B69D52b8b986AA4E0BAc6838Af6698``


.. image:: https://github.com/carlomazzaferro/scikit-hts/workflows/main%20workflow/badge.svg?branch=master
    :target: https://github.com/carlomazzaferro/scikit-hts/actions

.. image:: https://badge.fury.io/py/scikit-hts.svg
    :target: https://badge.fury.io/py/scikit-hts

.. image:: https://readthedocs.org/projects/racket/badge/?version=latest
    :target: https://racket.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


.. image:: https://codecov.io/gh/carlomazzaferro/scikit-hts/branch/master/graph/badge.svg?token=K4OAC8C51T
    :target: https://codecov.io/gh/carlomazzaferro/scikit-hts
    :alt: Coverage

.. image:: https://pepy.tech/badge/scikit-hts/month
     :target: https://pepy.tech/project/scikit-hts/month
     :alt: Downloads/Month

.. image:: https://img.shields.io/badge/join-us%20on%20slack-gray.svg?longCache=true&logo=slack&colorB=brightgreen
    :target: https://join.slack.com/t/scikit-hts/shared_invite/zt-d5is54bp-iOeagm7Jv68ZTkjk_zezrA
    :alt: Slack


* `MIT License`_
* Documentation: https://scikit-hts.readthedocs.io/en/latest/

.. _`MIT License`: https://github.com/carlomazzaferro/scikit-hts/blob/master/LICENSE

Overview
--------

Building on the excellent work by Hyndman [1]_, we developed this package in order to provide a python implementation
of general hierarchical time series modeling.


.. [1] `Forecasting Principles and Practice. Rob J Hyndman and George Athanasopoulos. Monash University, Australia <https://otexts.com/fpp2/>`_.

.. note:: **STATUS**: alpha. Active development, but breaking changes may come.


Features
--------

* Supported and tested on ``python 3.6``, ``python 3.7`` and ``python 3.8``
* Implementation of Bottom-Up, Top-Down, Middle-Out, Forecast Proportions, Average Historic Proportions,
  Proportions of Historic Averages and OLS revision methods
* Support for representations of hierarchical and grouped time series
* Support for a variety of underlying forecasting models, inlcuding: SARIMAX, ARIMA, Prophet, Holt-Winters
* Scikit-learn-like API
* Geo events handling functionality for geospatial data, including visualisation capabilities
* Static typing for a nice developer experience
* Distributed training & Dask integration: perform training and prediction in parallel or in a cluster with Dask

Examples
--------

You can find code usages here: https://github.com/carlomazzaferro/scikit-hts-examples

Roadmap
-------

* More flexible underlying modeling support
    * [P] AR, ARIMAX, VARMAX, etc
    * [P] Bring-Your-Own-Model
    * [P] Different parameters for each of the models
* Decoupling reconciliation methods from forecast fitting
    * [W] Enable to use the reconciliation methods with pre-fitted models

| **P**: Planned
| **W**: WIP

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

