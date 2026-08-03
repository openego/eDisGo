.. _contributing:

Contributing
============

Contributions to eDisGo are welcome. This page covers how to set up a development
environment, the code standards we follow, how to run the tests and how to build the
documentation.

Development installation
------------------------

Clone the repository and install it in editable mode with the development extras
into a fresh Python 3.11 virtual environment:

.. code-block:: bash

    git clone https://github.com/openego/eDisGo.git
    cd eDisGo
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[dev]"
    pre-commit install   # install the git pre-commit hooks

On Windows, create the environment from
`eDisGo_env_dev.yml <https://github.com/openego/eDisGo/blob/dev/eDisGo_env_dev.yml>`_
with conda first (see :ref:`installation`).

Code standards
--------------

* **pre-commit hooks** — formatting and linting (Ruff, which replaced flake8, isort,
  pyupgrade and Black) run on every commit; with the hooks installed you cannot push
  until they pass. Run them on demand with ``pre-commit`` (most issues are fixed
  automatically).
* **Tests** — all tests must pass and new code must come with tests.
* **Type hints** — add type hints to new functions; they are rendered into the API
  reference.
* **Docstrings** — document parameters and behaviour (NumPy/Google style, parsed by
  Napoleon). For ``@property`` pairs, document both getter and setter in the getter's
  docstring.
* **Documentation** — update the relevant ``doc/`` pages and add a ``whatsnew`` entry
  for user-visible changes.

Running the tests
-----------------

Tests use ``pytest``. Some tests are slow, require a local database, or only run on
Linux; enable them with the corresponding flags:

.. code-block:: bash

    python -m pytest tests/ --runslow --runlocal --runonlinux

For faster runs you can parallelise with ``-n`` (locally use a small number of
workers) and re-run flaky database tests:

.. code-block:: bash

    python -m pytest tests/ -n 3 --reruns 3 --reruns-delay 5 --runslow --html=report.html

Building the documentation
--------------------------

The documentation is built with Sphinx (the API reference is generated automatically
from the source by AutoAPI, and the tutorial notebooks are rendered by nbsphinx,
which requires `pandoc <https://pandoc.org>`_). From the repository root:

.. code-block:: bash

    sphinx-build -E -a -b html ./doc/ <outputdir>

To check external links:

.. code-block:: bash

    sphinx-build ./doc/ -b linkcheck -d _build/doctrees _build/html

Adding ``-n`` (nitpicky mode) additionally reports unresolved cross-references and
type hints.

Submitting changes
------------------

Develop on a feature branch and open a pull request against ``dev``. A PR can be
merged once the automated tests pass (in particular the Python 3.11 tests), the
branch is up to date with ``dev``, the pre-commit checks are green and a maintainer
has reviewed it. Please make sure new dependencies are added to ``setup.py`` and
``rtd_requirements.txt``, and new features are documented in ``whatsnew``.
