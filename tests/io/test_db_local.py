"""
Local-only tests for the egon-data database backend (``engine(ssh=True)``).

These tests exercise the local egon-data connection path that is otherwise
untested in CI (see issue #649): credential parsing, the SSH tunnel and the
dynamic ORM reflection in
:meth:`edisgo.tools.config.Config.import_tables_from_oep`. They also cover the
local-backend correctness fixes for reading the ``egon_etrago_*`` tables from
the ``grid`` schema (#668) and reflecting primary-key-less egon-data fact
tables (#669).

They require access to a (SSH-tunneled) local egon-data database and are
therefore skipped by default. Run them explicitly with::

    pytest --runlocal tests/io/test_db_local.py

The database config YAML is taken from the ``EGON_DATA_CONFIG`` environment
variable, falling back to ``~/.ssh/egon-data.configuration.yaml`` (see
``conftest.py``). The configured SSH key must be loadable (e.g. added to the
ssh-agent) for the tunnel to come up.
"""
import numpy as np
import pandas as pd
import pytest

from sqlalchemy import text

from edisgo.edisgo import EDisGo
from edisgo.io.db import credentials, session_scope_egon_data
from edisgo.tools.config import Config


@pytest.mark.local
class TestLocalEgonDataDB:
    def test_credentials(self):
        """Credentials YAML parses into the full local + SSH connection info."""
        cred = credentials(pytest.egon_data_config_yml)
        for key in (
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "PORT",
            "SSH_HOST",
            "SSH_USER",
            "SSH_PKEY",
            "PGRES_HOST",
        ):
            assert key in cred, f"missing '{key}' in parsed credentials"

    def test_engine_ssh_connects(self, engine_local):
        """The SSH-tunneled engine is a local psycopg2 engine and connects."""
        url = str(engine_local.url)
        assert "psycopg2" in url
        assert "openenergyplatform" not in url
        with engine_local.connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar() == 1

    def test_query_egon_table(self, engine_local):
        """At least one real data query against the local egon-data DB."""
        with engine_local.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM grid.egon_etrago_bus")
            ).scalar()
        assert count > 0

    def test_import_tables_reflection_grid_schema(self, engine_local):
        """
        Reflect ``egon_etrago_bus``/``egon_etrago_link`` from the ``grid``
        schema (the local-DB reflection branch + the #668 schema fix) and read
        a few rows.
        """
        config = Config()
        bus, link = config.import_tables_from_oep(
            engine_local, ["egon_etrago_bus", "egon_etrago_link"], "grid"
        )
        with session_scope_egon_data(engine_local) as session:
            df = pd.read_sql(session.query(bus).limit(5).statement, session.bind)
        assert not df.empty

    def test_import_tables_reflection_pkless(self, engine_local):
        """
        A primary-key-less egon-data fact table reflects without
        ``ArgumentError`` and is queryable (#669).
        """
        config = Config()
        (table,) = config.import_tables_from_oep(
            engine_local,
            ["egon_daily_heat_demand_per_climate_zone"],
            "demand",
        )
        with session_scope_egon_data(engine_local) as session:
            df = pd.read_sql(session.query(table).limit(5).statement, session.bind)
        assert not df.empty


@pytest.mark.local
class TestImportElectromobilityLocal:
    """
    Full end-to-end test of a real EDisGo data import against the local
    egon-data DB: :meth:`EDisGo.import_electromobility` loads electromobility
    data through the local backend (``engine(ssh=True)``) and processes it
    (charging demand allocation + integration of charging points into the
    grid).

    This mirrors ``tests/test_edisgo.py::test_import_electromobility_oedb``,
    which runs against the remote OEP, so the same egon-data dataset is
    expected and the assertions use the same reference values.
    """

    def test_import_electromobility_local_db(self, engine_local):
        edisgo = EDisGo(
            ding0_grid=pytest.ding0_test_network_3_path, legacy_ding0_grids=False
        )

        edisgo.import_electromobility(
            data_source="oedb", scenario="eGon2035", engine=engine_local
        )

        # --- data was loaded from the local DB ---
        assert len(edisgo.electromobility.charging_processes_df) == 324117
        assert edisgo.electromobility.eta_charging_points == 0.9

        # --- data was processed: charging demand allocated to charging parks ---
        total_charging_demand_at_charging_parks = sum(
            cp.charging_processes_df.chargingdemand_kWh.sum()
            for cp in list(edisgo.electromobility.potential_charging_parks)
            if cp.designated_charging_point_capacity > 0
        )
        total_charging_demand = (
            edisgo.electromobility.charging_processes_df.chargingdemand_kWh.sum()
        )
        assert np.isclose(
            total_charging_demand_at_charging_parks, total_charging_demand
        )

        # parks with allocated demand match the charging parks referenced in the
        # charging processes
        charging_park_ids = (
            edisgo.electromobility.charging_processes_df.charging_park_id.sort_values().unique()
        )
        potential_charging_parks_with_capacity = np.sort(
            [
                cp.id
                for cp in list(edisgo.electromobility.potential_charging_parks)
                if cp.designated_charging_point_capacity > 0.0
            ]
        )
        assert set(charging_park_ids) == set(potential_charging_parks_with_capacity)

        # --- charging points were integrated into the grid topology ---
        assert set(
            edisgo.electromobility.integrated_charging_parks_df.edisgo_id.sort_values().values
        ) == set(
            edisgo.topology.loads_df[
                edisgo.topology.loads_df.type == "charging_point"
            ].index.sort_values().values
        )
