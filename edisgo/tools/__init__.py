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

import importlib

from contextlib import contextmanager
from functools import cache

from saio import register_schema
from sqlalchemy.orm import Session


@cache
def _oep_engine():
    """
    Engine for the tables of the legacy open_eGo data model on the OEP.

    Uses the OEP token (see :func:`edisgo.io.db.engine`), so the connection does
    not count against the limit of open connections that all anonymous OEP
    clients share.

    """
    from edisgo.io.db import engine

    return engine(ssh=False)


@contextmanager
def session_scope():
    """Function to ensure that sessions are closed properly."""
    session = Session(bind=_oep_engine())
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def legacy_oep_table(schema, table):
    """
    ORM class of a table of the legacy open_eGo data model on the OEP.

    The table definition is reflected from the OEP with saio, using the engine
    of :func:`session_scope`.

    Parameters
    ----------
    schema : str
        Database schema of the table, e.g. 'supply'.
    table : str
        Name of the table, e.g. 'ego_dp_res_powerplant'.

    """
    # register the schema on every call: other imports register the same
    # schema name with a different engine, e.g. the local egon-data database
    register_schema(schema, _oep_engine())
    return getattr(importlib.import_module(f"saio.{schema}"), table)
