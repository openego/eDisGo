Notes to developers
===================

Installation
~~~~~~~~~~~~

Clone repository from `GitHub <https://github.com/openego/edisgo>`_ and install
in developer mode::

    pip3 install -e <path-to-repo>

The package `Dingo <https://github.com/openego/dingo>`_ is currently not
available as up-to-date release. Thus, we install it via a commit reference.
Therefore use the flag :code:`--process-dependency-links` ::

    pip3 install -e <path-to-repo> --process-dependency-links


Documentation
~~~~~~~~~~~~~

Build the docs locally by first setting up the sphinx environment with (executed
from top-level folder)

.. code-block:: bash

    sphinx-apidoc -f -o doc/api edisgo

And then you build the html docs on your computer with

.. code-block:: bash

    sphinx-build -E -a doc/ doc/_html