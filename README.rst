.. |PyPI| image:: https://img.shields.io/pypi/v/ringity.svg
    :target: https://pypi.org/project/ringity/
.. |Python| image:: https://img.shields.io/pypi/pyversions/ringity.svg
    :target: https://pypi.org/project/ringity/
.. |License| image:: https://img.shields.io/pypi/l/ringity.svg
    :target: https://github.com/ClusterDuck123/ringity/blob/master/LICENSE.txt
.. |DOI| image:: https://zenodo.org/badge/196970975.svg
    :target: https://doi.org/10.5281/zenodo.4908926
.. |Version| image:: https://img.shields.io/github/v/tag/ClusterDuck123/ringity?style=social
    :target: https://github.com/ClusterDuck123/ringity/tags

ringity
===========

|DOI| |License| |Version|

ringity is a Python package to analyze various data structures with respect to their ring structure.
The general theory of ring scores is developed in [1].

- **Source:** https://github.com/ClusterDuck123/ringity
- **Issues:** https://github.com/ClusterDuck123/ringity/issues
- **Docs:** “coming soon”

🚀 Quick Examples
--------------------------

Ring Score of Networks
^^^^^^^^^^^^^^^^^^^^^^

Calculate the ring score of a network:

.. code-block:: python

    import ringity as rng
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([(i%100,(i+1)%100) for i in range(100)])
    dgm = rng.pdiagram(G)   # constructs a persistence diagram from G
    dgm.ring_score()        # -> 1

A publication describing how ringity computes ring scores of networks is coming
soon! (see `Upcoming publications`_ below).

Ring Score of Point Clouds
^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculate the ring score of a point cloud as described in [2]:

.. code-block:: python

    import numpy as np
    import ringity as rng
    t = np.linspace(0, 2*np.pi, 100)
    X = np.array((np.cos(t),np.sin(t))).T
    rng.ring_score(X)     # -> 1

📖 References
--------------

[1] Youssef, M. K. (2026).
*On the Non-Existence of Model Structures and the Existence of Ring Scores:
Two Studies in Combinatorial Topology.*
PhD thesis, EPFL, Lausanne (thesis no. 12216).
DOI: `10.5075/epfl-thesis-12216 <https://doi.org/10.5075/epfl-thesis-12216>`_.
Part II develops the general theory of ring scores.

[2] Maggs, K., Youssef, M. K., Pulver, C., Isma, J., Nguyên, T. J., Arzt, M.,
Karthaus, W., Harrington, H. A., Hess, K., Dotto, G. P. (2025).
*Topology identifies concurrent cyclic processes in single-cell transcriptomics
and androgen receptor function.*
bioRxiv preprint.
DOI: `10.1101/2025.01.09.632214 <https://doi.org/10.1101/2025.01.09.632214>`_.

Upcoming publications
^^^^^^^^^^^^^^^^^^^^^^^^

Proper publications are in preparation for each of the following and will be
added here as they become available:

- the general theory of ring scores,
- ring scores for networks,
- ring scores for point clouds.

📦 Install
-----------

Install the latest version of ringity:

.. code-block:: console

   $ pip install ringity

👋 Used ringity in your work?
------------------------------

Used ringity in a paper, preprint, thesis, talk, or project?
We'd love to hear about it!

Let us know by opening an issue: https://github.com/ClusterDuck123/ringity/issues

🐞 Bugs
---------

Please report any bugs that you find `here <https://github.com/ClusterDuck123/ringity/issues>`_.
Or, even better, fork the repository on `GitHub <https://github.com/ClusterDuck123/ringity/>`_
and create a pull request. All inputs, suggestions and changes are more than welcome!

📄 License
-----------

MIT — see `LICENSE <./LICENSE.txt>`_.

📚 How to cite
---------------

To cite the software, use the Zenodo DOI:
`10.5281/zenodo.4908926 <https://doi.org/10.5281/zenodo.4908926>`_

Until the dedicated publications listed above are available, please cite the
PhD thesis [1] for the theory of ring scores and the bioRxiv preprint [2] for
ring scores of point clouds.
