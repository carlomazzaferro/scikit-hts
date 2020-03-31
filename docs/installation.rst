.. highlight:: shell

============
Installation
============

From PyPi
---------

.. code-block:: console

    $ pip install scikit-hts


With optional dependencies
--------------------------

Geo Utilities
^^^^^^^^^^^^^

This allows the usage of ``scikit-hts``'s geo handling capabilities. See more: :ref:`Geo Handling Capabilities`.

See more at

.. code-block:: console

    $ pip install scikit-hts[geo]



Facebook's Prophet Support
^^^^^^^^^^^^^^^^^^^^^^^^^^

This allows to train models using Facebook's `Prophet`_

.. code-block:: console

    $ pip install scikit-hts[prophet]



Auto-Arima
^^^^^^^^^^

This allows to train models using Alkaline-ml's excellent `auto arima implementation`_

.. code-block:: console

    $ pip install scikit-hts[auto-arima]



Distributed Training
^^^^^^^^^^^^^^^^^^^^

This allows to run distributed training with a local or remote Dask cluster

.. code-block:: console

    $ pip install scikit-hts[distributed]



Everything
^^^^^^^^^^

Install's all optional dependencies

.. code-block:: console

    $ pip install scikit-hts[all]




From sources
------------

The sources for scikit-hts can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/carlomazzaferro/scikit-hts

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/carlomazzaferro/scikit-hts/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/carlomazzaferro/scikit-hts
.. _tarball: https://github.com/carlomazzaferro/scikit-hts/tarball/master
.. _auto arima implementation: https://github.com/alkaline-ml/pmdarima
.. _Prophet: https://facebook.github.io/prophet/