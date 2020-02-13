##########
scikit-hts
##########

.. image:: https://travis-ci.org/carlomazzaferro/racket.svg?branch=master
    :target: https://travis-ci.org/carlomazzaferro/racket

.. image:: https://img.shields.io/pypi/v/racket.svg
    :target: https://pypi.python.org/pypi/racket

.. image:: https://readthedocs.org/projects/racket/badge/?version=latest
    :target: https://racket.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
                
.. image:: https://coveralls.io/repos/github/carlomazzaferro/scikit-hts/badge.svg?branch=master
    :target: https://coveralls.io/github/carlomazzaferro/scikit-hts?branch=master
    :alt: Coverage

.. image:: https://pepy.tech/badge/scikit-hts/month
     :target: https://pepy.tech/project/scikit-hts/month
     :alt: Downloads/Month


Hierarchical Time Series with a familiar API


* Free software: GNU General Public License v3
* Documentation: https://scikit-hts.readthedocs.io/en/latest/


Overview
--------

Building on the excellent work by Hyndman [1]_, we developed this package in order to provide a python implementation
of general hierarchical time series modeling.


.. [1] forecasting_: Forecasting Principles and Practice. Rob J Hyndman and George Athanasopoulos. Monash University, Australia: https://otexts.com/fpp2/

.. _forecasting:  https://otexts.com/fpp2/

.. note:: **STATUS**: alpha. Active development, but breaking changes may come.


Features
--------

* Implementation of Bottom-Up, Top-Down, Middle-Out, Forecast Proportions, Average Historic Proportions, Proportions of
Historic Averages and OLS revision methods
* Support for a variety of underlying forecasting models, inlcuding: SARIMAX, ARIMA, Prophet, Holt-Winters
* Scikit-learn-like API
* Geo events handling functionality for geospatial data, including visualisation capabilities
* Static typing for a nice developer experience



Roadmap
-------

* More flexible underlying modeling support


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

