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

from contextlib import contextmanager
from functools import cache

from sqlalchemy.orm import sessionmaker


@cache
def _session_factory():
    """
    Return a session factory for the OEP, created on first use.

    The legacy open_eGo tables (schemas ``supply`` and ``model_draft``, mapped
    by :mod:`egoio.db_tables`) are read from the OEP. The engine is built on
    the first query rather than at import time, so that importing eDisGo does
    not open a connection.

    """
    from edisgo.io.db import engine

    # ssh=False pins the source to the OEP: the legacy tables do not exist in
    # an egon-data database, which engine() would otherwise auto-detect.
    return sessionmaker(bind=engine(ssh=False))


@contextmanager
def session_scope():
    """Function to ensure that sessions are closed properly."""
    session = _session_factory()()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
