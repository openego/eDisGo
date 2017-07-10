Notes to developers
===================

Documentation
~~~~~~~~~~~~~

Build the docs locally by first setting up the sphinx environment with (executed
from top-level folder)

.. code-block:: bash

    sphinx-apidoc -f -o doc/api edisgo

And then you build the html docs on your computer with

.. code-block:: bash

    sphinx-build -E -a doc/ doc/_html