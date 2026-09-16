# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# Copyright (c) Reiner Lemoine Institut gGmbH
# Contributors are listed in the version control history:
# https://github.com/openego/eDisGo/
#
# Documentation: https://edisgo.readthedocs.io/
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
SQLAlchemy dialect for reading from the Open Energy Platform (OEP).

Importing this package registers the dialect, so that

.. code-block:: python

    create_engine("postgresql+oedialect://:<token>@openenergyplatform.org")

connects to the OEP. :func:`edisgo.io.db.engine` builds that URL.

This is a reduced fork of `oedialect
<https://github.com/OpenEnergyPlatform/oedialect>`_ (AGPL-3.0-or-later, see
``LICENSE`` in this directory), taken from commit ``9d1afba`` of its
``upgrade-sqlalchemy`` branch. eDisGo carries its own copy because the
released oedialect 0.1.1 only works with SQLAlchemy 1.3: it cannot even be
imported with SQLAlchemy 2.0, and SQLAlchemy 2.0 is required by pandas from
version 2.2 on, which in turn is required for NumPy 2. The fork differs from
its origin in that it

* implements the SELECT compiler and the schema reflection against the
  SQLAlchemy 2.0 interfaces,
* is read-only - eDisGo never writes to the OEP, so the DDL compiler and the
  INSERT, UPDATE and DELETE compilation of the original are dropped, and
* does not register a global compilation rule for ``WKBElement``, which in the
  original changed how geometries compile for every dialect in the process.

Should oedialect release a SQLAlchemy 2.0 compatible version, this package can
be dropped in favour of it again.
"""

from sqlalchemy.dialects import registry

registry.register("postgresql.oedialect", "edisgo.io.oedialect.dialect", "OEDialect")
