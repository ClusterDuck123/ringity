.. Warning:: This package is stil under construction.

ringity
=======

ringity is a Python package for the analysis of complex networks 
with respect to their global ring structur.

- **Website (including documentation):** Not available yet.
- **Source:** https://github.com/kiri93/ringity
- **Bug reports:** https://github.com/kiri93/ringity/issues
- **Contact Person:** mk.youssef@hotmail.com

Simple example
--------------

Calculate ring score as described in 
"*The Emergence of Ring Structures in Biological Networks*"[1]:

.. code:: python

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
and create a pull request. All input and changes are very welcome!

License
-------

Released under the MIT License (see `LICENSE.txt`)::

   Copyright (c) 2019 Markus K. Youssef <mk.youssef@hotmail.com>
