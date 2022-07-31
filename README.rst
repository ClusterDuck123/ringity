ringity
=======

|DOI| |license| |version|

ringity is a Python package to analyze various data structures with respect to their ring structure.

- **Source:** https://github.com/ClusterDuck123/ringity
- **Bug reports:** https://github.com/ClusterDuck123/ringity/issues
- **Contact Person:** mk.youssef@hotmail.com
- **Documentation:** Not available yet.

Notes
-----

This package is still under construction!

Simple network example
----------------------

Calculate ring score as described in [1]:

.. code:: python

    >>> import ringity as rng
    >>> import networkx as nx
    >>> G = nx.Graph()
    >>> G.add_edges_from([(i%100,(i+1)%100) for i in range(100)])
    >>> dgm = rng.diagram(G)
    >>> dgm.ring_score()
    1

[1]: Paper not available yet.

Simple point cloud example
--------------------------

Calculate ring score as described in [2]:

.. code:: python

    >>> import numpy as np
    >>> import ringity as rng
    >>> t = np.linspace(0, 2*np.pi, 100)
    >>> X = np.array((np.cos(t),np.sin(t))).T
    >>> rng.ring_score(X)
    1
    
[2]: Paper not available yet.

Install
-------

Install the latest version of ringity::

    $ pip install ringity

Latest Changes
--------------
- Changed foldername `data` to `_data`
- The persistent diagram class `PersistenceDiagram` has `ring_score` now as a method, not as a property. As a result, the function is now called via `dgm.ring_score()` insted of `dgm.ring_score`! (**Notice the change in brackets**) We appologize if this is breaking old code; such changes will be avoided in the future.
- Integration of various ring-score "flavours".
- Extended usage for point clouds.


Bugs
----

Please report any bugs that you find `here <https://github.com/ClusterDuck123/ringity/issues>`_.
Or, even better, fork the repository on `GitHub <https://github.com/ClusterDuck123/ringity/>`_
and create a pull request. All inputs, suggestions and changes are more than welcome!

License
-------

Released under the MIT License (see `LICENSE.txt`)::

   Copyright (c) 2019 Markus K. Youssef <mk.youssef@hotmail.com>

How to cite
-----------

If you want to cite this package, please use the DOI:
`10.5281/ZENODO.4908927 <https://doi.org/10.5281/ZENODO.4908927>`_


.. =================================
..         Badge definitions
.. =================================
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2FZENODO.4908927-orange
   :target: https://zenodo.org/badge/latestdoi/196970975
.. |license| image:: https://img.shields.io/github/license/kiri93/ringity
.. |version| image:: https://img.shields.io/github/v/tag/kiri93/ringity?style=social
