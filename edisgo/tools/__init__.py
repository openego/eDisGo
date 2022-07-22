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
