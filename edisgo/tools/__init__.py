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

from sqlalchemy.orm import Session


@cache
def _oep_engine():
    """
    Engine for the legacy (egoio) tables on the OEP.

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
