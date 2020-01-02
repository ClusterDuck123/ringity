ringity
=======

ringity is a Python package to analyze networks in their global ring structure.

- **Source:** https://github.com/kiri93/ringity
- **Bug reports:** https://github.com/kiri93/ringity/issues
- **Contact Person:** mk.youssef@hotmail.com
- **Documentation:** Not available yet.

Notes
-----

This package is still under construction.

Simple example
--------------

Calculate ring score as described in [1]:

.. code:: python

    >>> import ringity as rng
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edges_from([(i%100,(i+1)%100) for i in range(100)])
    >>> dgm = rng.diagram(G)
    >>> dgm.score
    1

[1]: Paper not available yet.

Install
-------

Install the latest version of ringity::

    $ pip install ringity

Bugs
----

Please report any bugs that you find `here <https://github.com/kiri93/ringity/issues>`_.
Or, even better, fork the repository on `GitHub <https://github.com/kiri93/ringity/>`_
and create a pull request. All inputs, suggestions and changes are more than welcome!

License
-------

Released under the MIT License (see `LICENSE.txt`)::

   Copyright (c) 2019 Markus K. Youssef <mk.youssef@hotmail.com>
