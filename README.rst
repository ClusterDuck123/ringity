.. |PyPI| image:: https://img.shields.io/pypi/v/ringity.svg
   :target: https://pypi.org/project/ringity/
.. |Python| image:: https://img.shields.io/pypi/pyversions/ringity.svg
   :target: https://pypi.org/project/ringity/
.. |License| image:: https://img.shields.io/pypi/l/ringity.svg
   :target: https://github.com/ClusterDuck123/ringity/blob/main/LICENSE
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4908927.svg
   :target: https://doi.org/10.5281/zenodo.4908927
.. |version| image:: https://img.shields.io/github/v/tag/ClusterDuck123/ringity?style=social

ringity
=======

|DOI| |license| |version|

ringity is a Python package to analyze various data structures with respect to their ring structure.

- **Source:** https://github.com/ClusterDuck123/ringity
- **Issues:** https://github.com/ClusterDuck123/ringity/issues
- **Docs:** “coming soon”

Simple network example
----------------------

Calculate ring score as described in [1]:

.. code-block:: python

    import ringity as rng
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([(i%100,(i+1)%100) for i in range(100)])
    dgm = rng.pdiagram(G)   # constructs a persistence diagram from G
    dgm.ring_score()        # -> 1

[1]: Paper not available yet.

Simple point cloud example
--------------------------

Calculate ring score as described in [2]:

.. code-block:: python

    import numpy as np
    import ringity as rng
    t = np.linspace(0, 2*np.pi, 100)
    X = np.array((np.cos(t),np.sin(t))).T
    rng.ring_score(X)     # -> 1

[2]: Paper not available yet.

Install
-------

Install the latest version of ringity:

.. code-block:: console

   $ pip install ringity

Latest Changes
--------------
- Changed folder name `data` to `_data`
- The persistence diagram class `PersistenceDiagram` has `ring_score` now as a method, not as a property. As a result, the function is now called via `dgm.ring_score()` instead of `dgm.ring_score`! (**Notice the change in brackets**) We apologize if this is breaking old code; such changes will be avoided in the future.
- Integration of various ring-score "flavours".
- Extended usage for point clouds.


Bugs
----

Please report any bugs that you find `here <https://github.com/ClusterDuck123/ringity/issues>`_.
Or, even better, fork the repository on `GitHub <https://github.com/ClusterDuck123/ringity/>`_
and create a pull request. All inputs, suggestions and changes are more than welcome!

License
-------

MIT — see `LICENSE <./LICENSE>`_.

How to cite
-----------

DOI: `10.5281/zenodo.4908927 <https://doi.org/10.5281/zenodo.4908927>`_
