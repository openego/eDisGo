.. _dev-notes:

Notes to developers
===================

Installation
------------

Clone the repository from `GitHub <https://github.com/openego/edisgo>`_ and change into
the eDisGo directory:

.. code-block:: bash

    cd eDisGo


Installation using Linux
~~~~~~~~~~~~~~~~~~~~~~~~

To set up a source installation using linux simply use a virtual environment and install
the source code with pip. Make sure to use python3.7 or higher (recommended
python3.8). **After** setting up your virtual environment and activating it run the
following commands within your eDisGo directory:

.. code-block:: bash

    python -m pip install -e .[full]  # install eDisGo from source
    pre-commit install  # install pre-commit hooks


Installation using Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~

For Windows users we recommend using Anaconda and to install the geo stack
using the conda-forge channel prior to installing eDisGo. You may use the provided
`eDisGo_env_dev.yml file <https://github.com/openego/eDisGo/blob/dev/eDisGo_env_dev.yml>`_
to do so. Create the virtual environment with:

.. code-block:: bash

    conda env create -f path/to/eDisGo_env_dev.yml  # install eDisGo from source

Activate the newly created environment and install the pre-commit hooks with:

.. code-block:: bash

    conda activate eDisGo_env_dev
    pre-commit install  # install pre-commit hooks

This will install eDisGo with all its dependencies.

Installation using MacOS
~~~~~~~~~~~~~~~~~~~~~~~~~

We don't have any experience with our package on MacOS yet! If you try eDisGo on MacOS
we would be happy if you let us know about your experience!


Code standards
--------------

* **pre-commit hooks**: Make sure to use the provided pre-commit hooks
* **pytest**: Make sure that all pytest tests are passing and add tests for every new code base
* **Documentation of `@property` functions**: Put documentation of getter and setter
  both in Docstring of getter, see
  `on Stackoverflow <https://stackoverflow.com/questions/16025462/what-is-the-right-way-to-put-a-docstring-on-python-property/16025754#16025754>`_
* Order of public/private/protected methods, property decorators, etc. in a class: TBD


Documentation
-------------

You can build the docs locally as follows (executed from top-level eDisGo directory):

.. code-block:: bash

    sphinx-build -E -a -b html ./doc/ <outputdir>

To manually check if external links in the documentation work, you can run the following command (internal links are not checked by this):

.. code-block:: bash

   sphinx-build ./doc/ -b linkcheck -d _build/doctrees _build/html

Internal links can be checked adding -n option when building the documentation. This will
also raise warnings for type hinting, so it is a bit confusing, but can still be helpful.

.. code-block:: bash

    sphinx-build -n -E -a -b html ./doc/ <outputdir>
