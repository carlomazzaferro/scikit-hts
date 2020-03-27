Supported Models
================

Scikit-hts extends the work done by Hyndman in a few ways. One of the most important
ones is the ability to use a variety of different underlying modeling techniques to
predict the base forecasts.

We have implemented so far 4 kinds of underlying models:

1. `Auto-Arima`_, thanks to the excellent implementation provided by the folks at alkaline-ml
2. `SARIMAX`_, implemented by the `statsmodels`_ package
3. `Holt-Winters`_ exponential smoothing, also implemented in `statsmodels`_
4. `Facebook's Prophet`_

The full feature set of the underlying models is supported, including exogenous
variables handling. Upon instantiation, use keyword arguments to pass the the
arguments you need to the underlying model instantiation, fitting, and prediction.


.. _`Auto-Arima`: https://github.com/alkaline-ml/pmdarima
.. _`statsmodels`: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
.. _`SARIMAX`: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
.. _`Holt-Winters`: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
.. _`Facebook's Prophet`: https://facebook.github.io/prophet/


.. note:: The main development focus is adding more support underlying models. Stay tuned, or feel free to check out the :ref:`Contribute` guide.



Models
------

.. automodule:: hts.model
    :members:

