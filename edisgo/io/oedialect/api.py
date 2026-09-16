# This file is part of eDisGo (Electrical Distribution Grid Optimization),
# a Python package for analyzing flexibility options in distribution grids.
#
# It is a fork of the "oedialect" package
# (https://github.com/OpenEnergyPlatform/oedialect), Copyright (c) Reiner
# Lemoine Institut gGmbH and the Open Energy Platform contributors, from which
# the wire format of the OEP API and the result processing are taken. See
# LICENSE in this directory and the module docstring of __init__.py.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
DB-API 2.0 style client for the Open Energy Platform (OEP) HTTP API.

The OEP does not expose its PostgreSQL server; queries are sent to
``https://openenergyplatform.org/api/v0/`` as JSON documents. This module
provides the connection and cursor objects the SQLAlchemy dialect in
:mod:`edisgo.io.oedialect.dialect` drives: they speak HTTP, but look like a
DB-API driver to SQLAlchemy.
"""

from __future__ import annotations

import copy
import datetime
import json
import logging
import os

from decimal import Decimal

import psycopg2
import requests
import sqlalchemy

from dateutil.parser import parse as parse_date
from psycopg2.extensions import PYINTERVAL
from shapely import wkb
from sqlalchemy.dialects.postgresql.base import _DECIMAL_TYPES

logger = logging.getLogger(__name__)

#: Host the API is served from. Requests to any of the historic host names are
#: sent here.
OEP_URL = "openenergyplatform.org"

#: Historic host names of the OEP, all served by :data:`OEP_URL` today.
OEP_ALIASES = frozenset(
    {
        "oep.iks.cs.ovgu.de",
        "oep2.iks.cs.ovgu.de",
        "oep.iws.cs.ovgu.de",
        "oep2.iws.cs.ovgu.de",
        "openenergyplatform.org",
        "openenergy-platform.org",
    }
)

#: Schema used when a table is referenced without one.
DEFAULT_SCHEMA = "sandbox"

# DB-API interface expected by SQLAlchemy. The exception hierarchy is taken
# from psycopg2 so that the PostgreSQL dialect classifies errors as usual.
apilevel = "2.0"
threadsafety = 2
paramstyle = "pyformat"

Error = psycopg2.Error
DatabaseError = psycopg2.DatabaseError
IntegrityError = psycopg2.IntegrityError
InterfaceError = psycopg2.InterfaceError
InternalError = psycopg2.InternalError
NotSupportedError = psycopg2.NotSupportedError
OperationalError = psycopg2.OperationalError
ProgrammingError = psycopg2.ProgrammingError


class ConnectionException(Exception):
    """Raised when the OEP API answers with an error status."""


class CursorError(Exception):
    """Raised when a cursor cannot be opened on the OEP API."""


def connect(dsn=None, connection_factory=None, cursor_factory=None, **kwargs):
    """DB-API entry point used by SQLAlchemy to open a connection."""
    return OEPConnection(**kwargs)


def _json_default(obj):
    """
    Serialise values the OEP API expects as typed JSON objects.

    Dates, times and decimals are sent as ``{"type": "value", ...}`` documents;
    anything else falls back to its string representation.
    """
    if isinstance(obj, datetime.datetime):
        return {"type": "value", "datatype": "datetime", "value": obj.isoformat()}
    elif isinstance(obj, datetime.date):
        return {"type": "value", "datatype": "date", "value": obj.isoformat()}
    elif isinstance(obj, datetime.time):
        return {"type": "value", "datatype": "time", "value": obj.isoformat()}
    elif isinstance(obj, Decimal):
        return {"type": "value", "datatype": "Decimal", "value": str(obj)}
    return str(obj)


def _check_response(response, content=None):
    """Turn an error status of an API response into an exception."""
    content = content or {}
    if 400 <= response.status_code < 500:
        raise ConnectionException(
            "HTTP {} ({}): {}".format(
                response.status_code, response.reason, content.get("reason", "")
            )
        )
    elif 500 <= response.status_code < 600:
        raise ConnectionException(
            "Server side error: " + content.get("reason", "No reason returned")
        )


class OEPConnection:
    """
    DB-API connection to the OEP API.

    Parameters
    ----------
    host : str
        Host name of the OEP.
    port : int
        Port to send the requests to. Port 80 is mapped to 443, as the API is
        only served over HTTPS.
    user : str
        Unused, kept for DB-API compatibility.
    database : str
        Unused, kept for DB-API compatibility.
    password : str
        OEP user token. Without one the API applies a low request quota, which
        makes concurrent queries fail.

    """

    def __init__(self, host=OEP_URL, port=80, user="", database="", password=""):
        self._host = host
        self._port = port
        self._user = user
        self._token = password
        self._cursors = set()
        self._id = self.post("advanced/connection/open", {})["content"]["connection_id"]

    @property
    def _base_url(self):
        protocol = os.environ.get("OEDIALECT_PROTOCOL", "https")
        if protocol not in ("http", "https"):
            raise ValueError(
                f"OEDIALECT_PROTOCOL must be 'http' or 'https', got '{protocol}'."
            )
        host = OEP_URL if self._host in OEP_ALIASES else self._host
        port = 443 if self._port == 80 else self._port
        return f"{protocol}://{host}:{port}/api/v0/"

    @property
    def _headers(self):
        if self._token:
            return {"Authorization": f"Token {self._token}"}
        return {}

    @property
    def _verify(self):
        return os.environ.get("OEDIALECT_VERIFY_CERTIFICATE", "TRUE") == "TRUE"

    def post(self, suffix, query, cursor_id=None, requires_connection_id=False):
        """
        Send one request to the API and return the decoded answer.

        Parameters
        ----------
        suffix : str
            API endpoint, e.g. ``"advanced/search"``.
        query : dict
            Query document. The key ``request_type`` selects the HTTP method,
            which defaults to POST.
        cursor_id : str or None
            Cursor the request belongs to.
        requires_connection_id : bool
            Whether the connection ID has to be sent along.

        Returns
        -------
        dict
            Decoded JSON answer.

        """
        request_type = query.get("request_type") if isinstance(query, dict) else None
        sender = {
            "put": requests.put,
            "delete": requests.delete,
            "get": requests.get,
        }.get(request_type, requests.post)

        query = {k: v for k, v in query.items() if k != "info_cache"}
        data = {"query": query}
        if requires_connection_id or cursor_id:
            data["connection_id"] = self._id
        if cursor_id:
            data["cursor_id"] = cursor_id

        response = sender(
            self._base_url + suffix,
            json=json.loads(json.dumps(data, default=_json_default)),
            headers=self._headers,
            verify=self._verify,
        )

        try:
            content = response.json()
        except ValueError:
            raise ConnectionException(f"Answer contains no JSON: {response!r}")

        _check_response(response, content)

        if isinstance(query, dict) and query.get("request_type") == "get":
            content = {"content": content}
        return content

    def post_expect_stream(self, suffix, query, cursor_id=None):
        """Send a request whose answer is streamed line by line."""
        data = {}
        if cursor_id:
            data["connection_id"] = self._id
            data["cursor_id"] = cursor_id

        response = requests.post(
            self._base_url + suffix,
            json=data,
            headers=self._headers,
            stream=True,
            verify=self._verify,
        )
        _check_response(response)

        for line in response.iter_lines():
            yield json.loads(line.decode("utf8").replace("'", '\\"'))

    def cursor(self, *args, **kwargs):
        cursor = OEPCursor(self)
        self._cursors.add(cursor)
        return cursor

    def close(self, *args, **kwargs):
        self.post("advanced/connection/close", {}, requires_connection_id=True)

    def commit(self, *args, **kwargs):
        self.post("advanced/connection/commit", {}, requires_connection_id=True)

    def rollback(self, *args, **kwargs):
        self.post("advanced/connection/rollback", {}, requires_connection_id=True)


def resolve_params(jsn, params):
    """
    Recursively substitute bind parameters into a query document.

    Parameters
    ----------
    jsn : dict or list or object
        Query document, or one of its elements.
    params : dict
        Bind parameters of the execution.

    Returns
    -------
    dict or list or object
        The document with the bind parameters filled in.

    """
    if jsn is None:
        return {"type": "value", "value": None}
    elif callable(jsn):
        # a bind parameter, see OECompiler.bindparam_string
        return jsn(params)
    elif isinstance(jsn, dict):
        return {k: resolve_params(v, params) for k, v in jsn.items()}
    elif isinstance(jsn, list):
        return [resolve_params(x, params) for x in jsn]
    elif params and isinstance(
        jsn,
        (
            str,
            sqlalchemy.sql.elements.quoted_name,
            sqlalchemy.sql.elements._truncated_label,
        ),
    ):
        return (jsn % params).strip("'<>").replace("'", '"')
    return jsn


class OEPCursor:
    """DB-API cursor executing compiled queries against the OEP API."""

    description = None
    rowcount = -1

    #: Result post-processing per PostgreSQL type OID.
    _cell_processors = {
        17: lambda cell: wkb.dumps(wkb.loads(cell, hex=True)),
        1114: lambda cell: parse_date(cell),
        1082: lambda cell: parse_date(cell).date(),
        1083: lambda cell: parse_date(cell).time(),
        1186: lambda cell: PYINTERVAL(cell, None),
    }

    def __init__(self, connection):
        self._connection = connection
        response = connection.post(
            "advanced/cursor/open", {}, requires_connection_id=True
        )
        if "content" not in response:
            raise CursorError(
                "Could not open cursor: " + response.get("reason", "No reason returned")
            )
        self._id = response["content"]["cursor_id"]

    def execute(self, query_obj, params=None):
        """
        Execute a query.

        Parameters
        ----------
        query_obj : dict or :class:`sqlalchemy.sql.compiler.Compiled`
            The query document, or a compiled statement holding one in its
            ``string`` attribute. Plain SQL strings are not supported by the
            API.
        params : dict or tuple or None
            Bind parameters to substitute into the query document.

        """
        if query_obj is None:
            return None
        if isinstance(query_obj, str):
            raise NotSupportedError(
                "The OEP API does not accept plain SQL strings. Build the query "
                "with SQLAlchemy instead."
            )
        query = query_obj if isinstance(query_obj, dict) else query_obj.string
        # SQLAlchemy caches compiled statements, so the query document may be
        # executed again with different parameters - substitute into a copy.
        query = copy.deepcopy(dict(query))
        requires_connection_id = query.get("requires_connection", False)
        query["connection_id"] = self._connection._id
        query["cursor_id"] = self._id

        # Always resolve: a statement whose only parameters are the values of an
        # IN clause is executed without any parameters, as SQLAlchemy would
        # substitute those into the statement itself.
        query = resolve_params(query, params or {})

        return self._execute_by_post(
            query.pop("command"), query, requires_connection_id=requires_connection_id
        )

    def executemany(self, query, params=None):
        if params is None:
            return self.execute(query)
        return [self.execute(query, p) for p in params]

    def _execute_by_post(self, command, query, requires_connection_id=False):
        response = self._connection.post(
            command,
            query,
            cursor_id=self._id,
            requires_connection_id=requires_connection_id,
        )
        result = response.get("content")
        if isinstance(result, dict):
            if "description" in result:
                self.description = result["description"]
            if "rowcount" in result:
                self.rowcount = result["rowcount"]
        return result

    def process_result(self, row):
        """Convert the JSON representation of a row to Python objects."""
        for i, column in enumerate(self.description):
            if not row[i]:
                continue
            type_oid = column[1]
            if type_oid in self._cell_processors:
                row[i] = self._cell_processors[type_oid](row[i])
            elif type_oid in _DECIMAL_TYPES:
                if isinstance(row[i], list):
                    row[i] = [Decimal(value) for value in row[i]]
                else:
                    row[i] = Decimal(row[i])
        return row

    def fetchone(self):
        row = self._connection.post(
            "advanced/cursor/fetch_one", {}, cursor_id=self._id
        )["content"]
        return self.process_result(row) if row else row

    def fetchall(self):
        # The rows are returned as a list, not a generator: SQLAlchemy closes
        # the cursor before it builds its rows from what fetchall() returned,
        # and the cursor is what streams them from the API.
        return [
            self.process_result(row)
            for row in self._connection.post_expect_stream(
                "advanced/cursor/fetch_all", {}, cursor_id=self._id
            )
        ]

    def fetchmany(self, size):
        return [
            self.process_result(row)
            for row in self._connection.post_expect_stream(
                "advanced/cursor/fetch_many", {"size": size}, cursor_id=self._id
            )
        ]

    def close(self):
        self._connection.post("advanced/cursor/close", {}, cursor_id=self._id)
