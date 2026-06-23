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

from __future__ import annotations

import importlib.util
import logging
import os
import re

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from geoalchemy2.types import Geometry
from sqlalchemy import create_engine, func
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.session import Session
from sshtunnel import SSHTunnelForwarder

if TYPE_CHECKING:
    from edisgo import EDisGo

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
    Return database connection parameters for the egon-data database.

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
    Initialize an SSH tunnel to a remote host according to the input arguments.
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
        ssh_pkey=cred["SSH_PKEY"],
        remote_bind_address=(cred["PGRES_HOST"], cred["PORT"]),
    )
    server.start()

    return str(server.local_bind_port)


def engine(
    path: Path | str = None, ssh: bool = False, token: Path | str = None
) -> Engine:
    """
    Engine for local or remote database.

    Parameters
    ----------
    path : str or pathlib.Path, optional (default=None)
        Path to configuration YAML file of egon-data database.
    ssh : bool (default=False)
        If False, connects to the remote Open Energy Platform database (using the
        token, see parameter `token`). If True, establishes an ssh tunnel to a local
        egon-data database using the connection information in the configuration YAML
        given through `path`.
    token : str or pathlib.Path, optional (default=None)
        Token for database connection or path to text file containing token.
        If empty the default token file in the config folder OEP_TOKEN.txt
        will be used. If the default token file is not found, no token
        will be used and the connection will be established without token.

    Returns
    -------
    :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine

    """

    if not ssh:
        # Github Actions token
        if "OEP_TOKEN" in os.environ:
            token = os.environ["OEP_TOKEN"]

            read = True
        else:
            read = False

            if token is None:
                spec = importlib.util.find_spec("edisgo")
                token = Path(spec.origin).resolve().parent / "config" / "OEP_TOKEN.txt"

            if token.is_file():
                logger.info(f"Getting OEP token from file {token}.")

                with open(token) as file:
                    token = file.read().strip()

                read = True

        database_url = "openenergyplatform.org"

        msg = ""

        if not read:
            msg = f"Token file {token} not found"
            token = ""
        # Check if the token format is valid
        elif not re.match(r"^[a-f0-9]{40}$", token):
            msg = (
                f"Invalid token format for token {token}. A 40 character "
                f"hexadecimal string was expected"
            )
            token = ""

        if msg:
            logger.warning(
                f"{msg}. Connecting to {database_url} without a user token. This may "
                f"cause connection errors due to connection limitations. Consider "
                f"setting up an OEP account and providing your user token."
            )

        return create_engine(
            f"postgresql+oedialect://:{token}@{database_url}",
            echo=False,
        )

    cred = credentials(path=path)
    local_port = ssh_tunnel(cred)

    return create_engine(
        f"postgresql+psycopg2://{cred['POSTGRES_USER']}:"
        f"{cred['POSTGRES_PASSWORD']}@{cred['PGRES_HOST']}:"
        f"{local_port}/{cred['POSTGRES_DB']}",
        echo=False,
    )


@contextmanager
def session_scope_egon_data(engine: Engine):
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


def sql_grid_geom(edisgo_obj: EDisGo) -> Geometry:
    """
    Returns the grid district geometry as a PostGIS geometry.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
        EDisGo object whose grid district geometry is used.

    Returns
    -------
    Geometry
        Grid district geometry built from its WKT and SRID, for use in spatial SQL
        queries.

    """
    return func.ST_GeomFromText(
        edisgo_obj.topology.grid_district["geom"].wkt,
        edisgo_obj.topology.grid_district["srid"],
    )


def get_srid_of_db_table(session: Session, geom_col: InstrumentedAttribute) -> int:
    """
    Returns the SRID of a geometry column in a database table.

    Parameters
    ----------
    session : Session
        SQLAlchemy session used to query the database.
    geom_col : InstrumentedAttribute
        Geometry column whose spatial reference identifier is determined.

    Returns
    -------
    int
        Spatial reference identifier (SRID) of the geometry column.

    """
    query = session.query(func.ST_SRID(geom_col)).limit(1)

    return pd.read_sql(sql=query.statement, con=query.session.bind).iat[0, 0]


def sql_within(geom_a: Geometry, geom_b: Geometry, srid: int):
    """
    Checks if geometry a is completely within geometry b.

    Parameters
    ----------
    geom_a : Geometry
        Geometry within `geom_b`.
    geom_b : Geometry
        Geometry containing `geom_a`.
    srid : int
        SRID geometries are transformed to in order to use the same SRID for both
        geometries.

    """
    return func.ST_Within(
        func.ST_Transform(
            geom_a,
            srid,
        ),
        func.ST_Transform(
            geom_b,
            srid,
        ),
    )


def sql_intersects(geom_col: InstrumentedAttribute, geom_shape: Geometry, srid: int):
    """
    Checks if a geometry column intersects a given geometry.

    Parameters
    ----------
    geom_col : InstrumentedAttribute
        Geometry column to test for intersection.
    geom_shape : Geometry
        Geometry to test the column against.
    srid : int
        SRID both geometries are transformed to before the intersection test.

    """
    return func.ST_Intersects(
        func.ST_Transform(
            geom_col,
            srid,
        ),
        func.ST_Transform(
            geom_shape,
            srid,
        ),
    )
