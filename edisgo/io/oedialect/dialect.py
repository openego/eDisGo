# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# It is a fork of the "oedialect" package
# (https://github.com/OpenEnergyPlatform/oedialect), Copyright (c) Reiner
# Lemoine Institut gGmbH and the Open Energy Platform contributors, from which
# the API endpoints used for reflection are taken. See LICENSE in this
# directory and the module docstring of __init__.py.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
SQLAlchemy dialect for the Open Energy Platform (OEP).

The dialect connects to the OEP's HTTP API instead of a PostgreSQL server:
statements are compiled to JSON documents (see
:mod:`edisgo.io.oedialect.compiler`) and sent through the client in
:mod:`edisgo.io.oedialect.api`. Schema reflection uses the API's
``advanced/get_*`` endpoints, because the ``pg_catalog`` queries the
PostgreSQL dialect would otherwise run are not available over the API.
"""

from __future__ import annotations

import logging

import geoalchemy2
import shapely

from sqlalchemy import util
from sqlalchemy.dialects.postgresql.base import PGDialect, PGExecutionContext
from sqlalchemy.engine import reflection
from sqlalchemy.engine.default import DefaultDialect

from edisgo.io.oedialect import api
from edisgo.io.oedialect.compiler import OECompiler, OETypeCompiler

logger = logging.getLogger(__name__)


class OEExecutionContext(PGExecutionContext):
    """Execution context that skips the server-side features the API lacks."""

    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor()


class OEDialect(PGDialect):
    """
    Read-only PostgreSQL dialect talking to the OEP API.

    The dialect is registered as ``postgresql+oedialect`` when
    :mod:`edisgo.io.oedialect` is imported, so an engine is created with
    ``create_engine("postgresql+oedialect://:<token>@openenergyplatform.org")``.
    """

    name = "oedialect"
    driver = "oedialect"

    statement_compiler = OECompiler
    type_compiler_cls = OETypeCompiler
    execution_ctx_cls = OEExecutionContext

    # A compiled statement is a JSON document that the cursor fills in with the
    # bind parameters of the execution; it is not re-used across executions.
    supports_statement_cache = False

    supports_comments = False
    supports_sane_rowcount = False
    supports_sane_multi_rowcount = False
    _supports_create_index_concurrently = False
    _supports_drop_index_concurrently = False

    # The multi-table reflection of the PostgreSQL dialect reads pg_catalog,
    # which the API does not expose. SQLAlchemy's generic implementations call
    # the single-table methods below, which use the API instead.
    get_multi_columns = DefaultDialect.get_multi_columns
    get_multi_pk_constraint = DefaultDialect.get_multi_pk_constraint
    get_multi_foreign_keys = DefaultDialect.get_multi_foreign_keys
    get_multi_indexes = DefaultDialect.get_multi_indexes
    get_multi_unique_constraints = DefaultDialect.get_multi_unique_constraints
    get_multi_check_constraints = DefaultDialect.get_multi_check_constraints
    get_multi_table_comment = DefaultDialect.get_multi_table_comment
    get_multi_table_options = DefaultDialect.get_multi_table_options

    def __init__(self, *args, **kwargs):
        # The API returns JSON, so values are passed through unchanged instead
        # of being serialised into a JSON string.
        kwargs["json_serializer"] = lambda x: x
        kwargs["json_deserializer"] = lambda x: x
        super().__init__(*args, **kwargs)
        self.default_schema_name = "model_draft"

    @classmethod
    def import_dbapi(cls):
        return api

    def initialize(self, connection):
        # PGDialect.initialize() determines the server's capabilities with
        # queries against pg_catalog, which the API does not answer.
        self.server_version_info = self._get_server_version_info(connection)

    def _get_server_version_info(self, connection):
        return (9, 3)

    def _get_default_schema_name(self, connection):
        return self.default_schema_name

    def on_connect(self):
        return None

    def do_ping(self, dbapi_connection):
        return True

    def _api(self, connection):
        """Return the API client behind a SQLAlchemy connection."""
        dbapi_connection = connection.connection
        return getattr(dbapi_connection, "driver_connection", dbapi_connection)

    def _post(self, connection, command, **query):
        """Send a request to the API and return its content."""
        return self._api(connection).post(command, dict(query, command=command))[
            "content"
        ]

    @staticmethod
    def _schema_query(schema, **query):
        if schema:
            query["schema"] = schema
        return query

    def has_schema(self, connection, schema, **kw):
        return self._post(connection, "advanced/has_schema", schema=schema)

    def has_table(self, connection, table_name, schema=None, **kw):
        return self._post(
            connection,
            "advanced/has_table",
            **self._schema_query(schema or api.DEFAULT_SCHEMA, table=table_name),
        )

    @reflection.cache
    def get_schema_names(self, connection, **kw):
        return self._post(connection, "advanced/get_schema_names")

    @reflection.cache
    def get_table_names(self, connection, schema=None, **kw):
        return self._post(
            connection, "advanced/get_table_names", **self._schema_query(schema)
        )

    @reflection.cache
    def get_view_names(self, connection, schema=None, **kw):
        return self._post(
            connection, "advanced/get_view_names", **self._schema_query(schema)
        )

    @reflection.cache
    def get_view_definition(self, connection, view_name, schema=None, **kw):
        return self._post(
            connection,
            "advanced/get_view_definition",
            **self._schema_query(schema, view_name=view_name),
        )

    @reflection.cache
    def get_columns(self, connection, table_name, schema=None, **kw):
        content = self._post(
            connection,
            "advanced/get_columns",
            **self._schema_query(schema, table=table_name),
        )
        domains = self._reflected_domains(content.get("domains") or {})
        enums = self._reflected_enums(content.get("enums") or {})

        columns = []
        for name, format_type, default, notnull, _attnum, _oid in content["columns"]:
            columns.append(
                {
                    "name": name,
                    "type": self._reflect_type(
                        format_type,
                        domains,
                        enums,
                        type_description=f"column '{name}'",
                        collation=None,
                    ),
                    "nullable": not notnull,
                    "default": default,
                    "autoincrement": False,
                    "comment": None,
                }
            )
        return columns

    @staticmethod
    def _reflected_domains(domains):
        """
        Bring the API's domain descriptions into the shape SQLAlchemy expects.

        The API returns ``{"<schema>.<name>": {"attype": ..., "nullable": ...,
        "default": ...}}``, while :meth:`~sqlalchemy.dialects.postgresql.base.
        PGDialect._reflect_type` looks domains up by their parsed, qualified
        name.
        """
        reflected = {}
        for qualified_name, domain in domains.items():
            schema, _, name = qualified_name.rpartition(".")
            reflected[tuple(util.quoted_token_parser(qualified_name))] = {
                "name": name,
                "schema": schema,
                "visible": False,
                "type": domain["attype"],
                "collation": None,
                "default": domain["default"],
                "nullable": domain["nullable"],
                "constraints": [],
            }
        return reflected

    @staticmethod
    def _reflected_enums(enums):
        """Bring the API's enum descriptions into the shape SQLAlchemy expects."""
        reflected = {}
        for qualified_name, labels in enums.items():
            schema, _, name = qualified_name.rpartition(".")
            reflected[tuple(util.quoted_token_parser(qualified_name))] = {
                "name": name,
                "schema": schema,
                "visible": False,
                "labels": list(labels),
            }
        return reflected

    @reflection.cache
    def get_pk_constraint(self, connection, table_name, schema=None, **kw):
        return self._post(
            connection,
            "advanced/get_pk_constraint",
            **self._schema_query(schema, table=str(table_name)),
        )

    @reflection.cache
    def get_foreign_keys(self, connection, table_name, schema=None, **kw):
        return self._post(
            connection,
            "advanced/get_foreign_keys",
            **self._schema_query(schema, table=table_name),
        )

    @reflection.cache
    def get_indexes(self, connection, table_name, schema=None, **kw):
        return self._post(
            connection,
            "advanced/get_indexes",
            **self._schema_query(schema, table=table_name),
        )

    @reflection.cache
    def get_unique_constraints(self, connection, table_name, schema=None, **kw):
        return self._post(
            connection,
            "advanced/get_unique_constraints",
            **self._schema_query(schema, table=table_name),
        )

    @reflection.cache
    def get_check_constraints(self, connection, table_name, schema=None, **kw):
        # Not offered by the API; an empty list keeps reflection working.
        return []

    @reflection.cache
    def get_table_comment(self, connection, table_name, schema=None, **kw):
        return {"text": None}

    def get_isolation_level(self, dbapi_connection):
        return "READ COMMITTED"

    def set_isolation_level(self, dbapi_connection, level):
        if level not in ("READ COMMITTED", "AUTOCOMMIT"):
            raise NotImplementedError(
                f"The OEP API does not support the isolation level {level}."
            )


# The API sends geometries as hex-encoded WKB strings, which GeoAlchemy2 only
# accepts as bytes. Patching the element rather than the result processor keeps
# geometry columns usable wherever they surface. Binary data coming from a
# direct PostgreSQL connection is passed through untouched.
_original_wkb_element_init = geoalchemy2.WKBElement.__init__


def _wkb_element_init(self, data, *args, **kwargs):
    if isinstance(data, str):
        data = shapely.wkb.dumps(shapely.wkb.loads(data, hex=True))
    _original_wkb_element_init(self, data, *args, **kwargs)


if geoalchemy2.WKBElement.__init__ is _original_wkb_element_init:
    geoalchemy2.WKBElement.__init__ = _wkb_element_init
