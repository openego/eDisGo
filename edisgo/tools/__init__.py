import os
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

if "READTHEDOCS" not in os.environ:
    from egoio.tools.db import connection

    Session = sessionmaker(bind=connection(readonly=True))


@contextmanager
def session_scope():
    """Function to ensure that sessions are closed properly."""
    session = Session()
    try:
        yield session
    except:
        session.rollback()
        raise
    finally:
        session.close()
