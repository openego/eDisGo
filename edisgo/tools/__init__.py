from egoio.tools.db import connection
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

#Session = sessionmaker(bind=connection(readonly=True))


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
