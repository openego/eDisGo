.. _dev-notes:

Notes to developers
===================

Installation
~~~~~~~~~~~~

Clone repository from `GitHub <https://github.com/openego/edisgo>`_ and install
in developer mode::

    pip3 install -e <path-to-repo>


Code style
~~~~~~~~~~

* **Documentation of `@property` functions**: Put documentation of getter and
    setter both in Docstring of getter, see
    `on Stackoverflow <https://stackoverflow.com/a/16025754/6385207>`_
* Order of public/private/protected methods, property decorators, etc. in a
    class: TBD


Documentation
~~~~~~~~~~~~~

Build the docs locally by first setting up the sphinx environment with (executed
from top-level folder)

.. code-block:: bash

    sphinx-apidoc -f -o doc/api edisgo

And then you build the html docs on your computer with

.. code-block:: bash

    sphinx-build -E -a doc/ doc/_html