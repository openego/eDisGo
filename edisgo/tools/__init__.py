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

import os

from contextlib import contextmanager

from sqlalchemy.orm import sessionmaker

if "READTHEDOCS" not in os.environ:
    from egoio.tools.db import connection

    Session = sessionmaker(bind=connection(readonly=True))


@contextmanager
def session_scope():
    """Function to ensure that sessions are closed properly."""
    session = Session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
