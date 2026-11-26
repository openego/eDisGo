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

Tests use ``pytest``. Tests that access the live Open Energy Platform (OEP) are
marked with ``oep`` and skipped before fixture setup unless ``--runoep`` is
supplied. To run the regular suite:

.. code-block:: bash

    python -m pytest tests/

No ``-m "not oep"`` filter is needed for regular runs: OEP tests are already
skipped unless ``--runoep`` is supplied. The optional filter reports them as
deselected instead of skipped. Some regular tests are slow or only run on Linux;
enable them with the corresponding flags (use ``--runonlinux`` only on Linux):

.. code-block:: bash

    python -m pytest tests/ --runslow --runonlinux

For faster runs you can parallelise with ``-n`` (locally use a small number of
workers):

.. code-block:: bash

    python -m pytest tests/ -n 3 --runslow --html=report.html

See ``tests/README.md`` for the test markers and how to write live OEP tests.

Run live tests separately, without parallel pytest workers or automatic retries:

.. code-block:: bash

    python -m pytest tests/ --runoep --runslow -m oep

Include ``--runonlinux`` on Linux for the full platform-appropriate selection.
``--runoep`` permits live tests; ``-m oep`` selects only those tests. Selecting
``-m oep`` or supplying ``--runslow`` alone does not enable OEP tests.

Live OEP tests need an OEP token, see :ref:`OEP token <oep-token>` in the data
sources guide. The ``oep_engine`` fixture checks token presence and the expected
format (40 lowercase hexadecimal characters) before creating the engine. It reads
``OEP_TOKEN`` when set, otherwise the package's ``edisgo/config/OEP_TOKEN.txt``.
A set environment variable overrides the file, including when its value is empty
or malformed. The token value is never included in the fixture's error message.

Legacy ``session_scope()`` connections also use eDisGo's token-aware OEP engine.
They do not request the test fixture and therefore bypass its precheck. Notebook
tests create their engines in a separate kernel and also bypass the fixture.
Do not commit tokens in source files, notebooks or configuration.

CI runs the three basic matrix entries with ``-m "not oep"`` and without the OEP
token. The Linux/Python 3.12 coverage entry runs regular and live tests together
using ``--runoep --runslow --runonlinux`` without a marker filter, then uploads
the combined measurement to Coveralls. It checks the ``OEP_TOKEN`` repository
secret before setup and has a 140-minute job timeout. There is no separate OEP
or coverage-merging job. An OEP failure fails the coverage entry and prevents
the upload; the basic entries remain independent.

OEP-enabled coverage entries share a concurrency group across branches, which
serializes the entire coverage entry, including setup and regular tests. Basic
entries can still run in parallel. New pushes to the same branch still cancel
outdated workflow runs. Dependabot executes only regular tests in all four
entries, without an OEP token or Coveralls upload. The job timeout is not a
database/request timeout.

Excluding OEP tests does not prohibit all network access. The plotting notebook
test uses the checked-out grid in ``tests/data/ding0_test_network_1`` without
downloading it. Standalone notebook use retains a GitHub download fallback when
the repository data is unavailable.

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
