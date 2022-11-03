from __future__ import annotations

import logging

from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder

logger = logging.getLogger(__name__)


def config_settings(path: Path | str) -> dict[str, dict[str, str | int | Path]]:
    """
    Return a nested dictionary containing the configuration settings.

    It's a nested dictionary because the top level has command names as keys
    and dictionaries as values where the second level dictionary has command
    line switches applicable to the command as keys and the supplied values
    as values.

    So you would obtain the ``--database-name`` configuration setting used
    by the current invocation of ``egon-data`` via

    .. code-block:: python

        settings()["egon-data"]["--database-name"]

    Parameters
    ----------
    path : pathlib.Path or str
        Path to configuration YAML file of egon-data database.

    Returns
    -------
    dict
        Nested dictionary containing the egon-data and optional ssh tunnel configuration
        settings.

    """
    if isinstance(path, str):
        path = Path(path)

    if not path.is_file():
        raise ValueError(f"Configuration file {path} not found.")
    with open(path) as f:
        return yaml.safe_load(f)


def credentials(path: Path | str) -> dict[str, str | int | Path]:
    """
    Return local database connection parameters.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to configuration YAML file of egon-data database.

    Returns
    -------
    dict
        Complete DB connection information.

    """
    translated = {
        "--database-name": "POSTGRES_DB",
        "--database-password": "POSTGRES_PASSWORD",
        "--database-host": "HOST",
        "--database-port": "PORT",
        "--database-user": "POSTGRES_USER",
    }
    configuration = config_settings(path=path)

    egon_config = configuration["egon-data"]

    update = {
        translated[flag]: egon_config[flag]
        for flag in egon_config
        if flag in translated
    }

    if "PORT" in update.keys():
        update["PORT"] = int(update["PORT"])

    egon_config.update(update)

    if "ssh-tunnel" in configuration.keys():
        translated = {
            "ssh-host": "SSH_HOST",
            "ssh-user": "SSH_USER",
            "ssh-pkey": "SSH_PKEY",
            "pgres-host": "PGRES_HOST",
        }

        update = {
            translated[flag]: configuration["ssh-tunnel"][flag]
            for flag in configuration["ssh-tunnel"]
            if flag in translated
        }

        egon_config.update(update)

    if "SSH_PKEY" in egon_config.keys():
        egon_config["SSH_PKEY"] = Path(egon_config["SSH_PKEY"]).expanduser()

        if not egon_config["SSH_PKEY"].is_file():
            raise ValueError(f"{egon_config['SSH_PKEY']} is not a file.")

    return egon_config


def ssh_tunnel(cred: dict) -> str:
    """
    Initialize a SSH tunnel to a remote host according to the input arguments.
    See https://sshtunnel.readthedocs.io/en/latest/ for more information.

    Parameters
    ----------
    cred : dict
        Complete DB connection information.

    Returns
    -------
    str
        Name of local port.

    """
    server = SSHTunnelForwarder(
        ssh_address_or_host=(cred["SSH_HOST"], 22),
        ssh_username=cred["SSH_USER"],
        ssh_private_key=cred["SSH_PKEY"],
        remote_bind_address=(cred["PGRES_HOST"], cred["PORT"]),
    )
    server.start()

    return str(server.local_bind_port)


def engine(path: Path | str, ssh: bool = False) -> Engine:
    """
    Engine for local or remote database.

    Parameters
    ----------
    path : dict
        Path to configuration YAML file of egon-data database.
    ssh : bool
        If True try to establish ssh tunnel from given information within the
        configuration YAML. If False try to connect to local database.

    Returns
    -------
    sqlalchemy.engine.base.Engine
        Database engine

    """
    cred = credentials(path=path)

    if not ssh:
        return create_engine(
            f"postgresql+psycopg2://{cred['POSTGRES_USER']}:"
            f"{cred['POSTGRES_PASSWORD']}@{cred['HOST']}:"
            f"{cred['PORT']}/{cred['POSTGRES_DB']}",
            echo=False,
        )

    local_port = ssh_tunnel(cred)

    return create_engine(
        f"postgresql+psycopg2://{cred['POSTGRES_USER']}:"
        f"{cred['POSTGRES_PASSWORD']}@{cred['PGRES_HOST']}:"
        f"{local_port}/{cred['POSTGRES_DB']}",
        echo=False,
    )


def select_dataframe(sql, db_engine, index_col=None):
    """Select data from local database as pandas.DataFrame

    Parameters
    ----------
    sql : str
        SQL query to be executed.
    db_engine : sqlalchemy.engine.base.Engine
        Database engine
    index_col : str, optional
        Column(s) to set as index(MultiIndex). The default is None.

    Returns
    -------
    df : pandas.DataFrame
        Data returned from SQL statement.

    """

    df = pd.read_sql(sql, db_engine, index_col=index_col)

    if df.size == 0:
        logger.warning(f"No data returned by statement:\n{sql}")

    return df


def select_geodataframe(sql, db_engine, index_col=None, geom_col="geom", epsg=3035):
    """Select data from local database as geopandas.GeoDataFrame

    Parameters
    ----------
    sql : str
        SQL query to be executed.
    db_engine : sqlalchemy.engine.base.Engine
        Database engine
    index_col : str, optional
        Column(s) to set as index(MultiIndex). The default is None.
    geom_col : str, optional
        column name to convert to shapely geometries. The default is 'geom'.
    epsg : int, optional
        EPSG code specifying output projection. The default is 3035.

    Returns
    -------
    gdf : pandas.DataFrame
        Data returned from SQL statement.

    """

    gdf = gpd.read_postgis(sql, db_engine, index_col=index_col, geom_col=geom_col)

    if gdf.size == 0:
        logger.warning(f"No data returned by statement:\n{sql}")

    return gdf.to_crs(epsg=epsg)


@contextmanager
def session_scope(engine: Engine):
    """Provide a transactional scope around a series of operations."""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:  # noqa: E722
        session.rollback()
        raise
    finally:
        session.close()
