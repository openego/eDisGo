import os
import socket
import textwrap

import pytest

from pathlib import Path

from sshtunnel import SSHTunnelForwarder


def _get_egon_config():
    """Resolve egon-data YAML config path (same logic as conftest.py)."""
    return os.environ.get(
        "EGON_DATA_CONFIG",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "egon-data.configuration.yaml",
        ),
    )


def _port_is_open(port: int) -> bool:
    """Check if a local TCP port is open."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class TestSSHTunnel:
    """Tests for SSH tunnel lifecycle management in edisgo.io.db."""

    @pytest.mark.local
    def test_ssh_tunnel_returns_server(self):
        """ssh_tunnel() returns a (port, server) tuple with an active tunnel."""
        from edisgo.io.db import credentials, ssh_tunnel

        cred = credentials(path=_get_egon_config())
        if "SSH_HOST" not in cred:
            pytest.skip("No SSH config in YAML")

        port, server = ssh_tunnel(cred)
        try:
            assert isinstance(server, SSHTunnelForwarder)
            assert isinstance(port, str)
            assert server.is_active
            assert _port_is_open(int(port))
        finally:
            server.stop()

    @pytest.mark.local
    def test_engine_stores_ssh_server(self):
        """engine(path=...) stores the SSH server on engine._ssh_server."""
        from edisgo.io.db import engine

        eng = engine(path=_get_egon_config())
        try:
            assert hasattr(eng, "_ssh_server")
            if eng._ssh_server is not None:
                assert isinstance(eng._ssh_server, SSHTunnelForwarder)
                assert eng._ssh_server.is_active
        finally:
            if eng._ssh_server is not None:
                eng._ssh_server.stop()
            eng.dispose()

    @pytest.mark.local
    def test_ssh_server_cleanup(self):
        """After stop(), the tunnel is no longer active and the port is closed."""
        from edisgo.io.db import engine

        eng = engine(path=_get_egon_config())
        ssh_server = eng._ssh_server
        if ssh_server is None:
            eng.dispose()
            pytest.skip("No SSH tunnel used for this config")

        port = ssh_server.local_bind_port
        assert ssh_server.is_active
        assert _port_is_open(port)

        eng.dispose()
        ssh_server.stop()

        assert not ssh_server.is_active
        assert not _port_is_open(port)

    def test_engine_without_ssh_has_no_server(self):
        """OEP engine has _ssh_server = None (no tunnel)."""
        from edisgo.io.db import engine

        eng = engine()
        assert not hasattr(eng, "_ssh_server") or eng._ssh_server is None
        eng.dispose()


class TestConfigSettings:
    """Tests for config_settings() error handling."""

    def test_file_not_found(self):
        from edisgo.io.db import config_settings

        with pytest.raises(ValueError, match="not found"):
            config_settings("/nonexistent/path/config.yaml")

    def test_file_not_found_str_path(self):
        """String path is converted to Path and validated."""
        from edisgo.io.db import config_settings

        with pytest.raises(ValueError, match="not found"):
            config_settings("nonexistent.yaml")


class TestCredentials:
    """Tests for credentials() validation."""

    def test_invalid_ssh_pkey(self, tmp_path):
        from edisgo.io.db import credentials

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            egon-data:
              --database-name: testdb
              --database-password: testpw
              --database-host: localhost
              --database-port: 5432
              --database-user: testuser
            ssh-tunnel:
              ssh-host: example.com
              ssh-user: sshuser
              ssh-pkey: /nonexistent/key.pem
              pgres-host: dbhost
        """))

        with pytest.raises(ValueError, match="is not a file"):
            credentials(path=yaml_file)


class TestEngineOEP:
    """Tests for OEP engine token handling."""

    def test_env_token(self, monkeypatch):
        """Engine uses OEP_TOKEN from environment."""
        from edisgo.io.db import engine

        valid_token = "a" * 40
        monkeypatch.setenv("OEP_TOKEN", valid_token)
        eng = engine()
        assert f":{valid_token}@" in str(eng.url)
        eng.dispose()

    def test_token_file_not_found(self, monkeypatch, caplog):
        """Missing token file logs a warning and connects without token."""
        import logging

        from edisgo.io.db import engine

        monkeypatch.delenv("OEP_TOKEN", raising=False)

        with caplog.at_level(logging.WARNING, logger="edisgo.io.db"):
            eng = engine(token=Path("/nonexistent/OEP_TOKEN.txt"))

        assert "not found" in caplog.text
        eng.dispose()

    def test_invalid_token_format(self, monkeypatch, tmp_path, caplog):
        """Invalid token format logs a warning."""
        import logging

        from edisgo.io.db import engine

        monkeypatch.delenv("OEP_TOKEN", raising=False)

        token_file = tmp_path / "OEP_TOKEN.txt"
        token_file.write_text("not-a-valid-hex-token")

        with caplog.at_level(logging.WARNING, logger="edisgo.io.db"):
            eng = engine(token=token_file)

        assert "Invalid token format" in caplog.text
        # Invalid token should be replaced with empty string
        assert ":@" in str(eng.url)
        eng.dispose()


class TestSessionScope:
    """Tests for session_scope_egon_data() exception handling."""

    def test_rollback_on_exception(self):
        """Session rolls back and re-raises on exception."""
        from sqlalchemy import Column, Integer, String, create_engine
        from sqlalchemy.ext.declarative import declarative_base

        from edisgo.io.db import session_scope_egon_data

        eng = create_engine("sqlite:///:memory:")
        Base = declarative_base()

        class DummyTable(Base):
            __tablename__ = "dummy"
            id = Column(Integer, primary_key=True)
            name = Column(String)

        Base.metadata.create_all(eng)

        # Insert a row, then force an exception — row should be rolled back
        with pytest.raises(RuntimeError, match="forced"):
            with session_scope_egon_data(eng) as session:
                session.add(DummyTable(id=1, name="test"))
                raise RuntimeError("forced error")

        # Verify rollback: table should be empty
        with session_scope_egon_data(eng) as session:
            count = session.query(DummyTable).count()
            assert count == 0

        eng.dispose()


class TestSqlFunctions:
    """Tests for SQL helper functions."""

    def test_sql_within_returns_binary_expression(self):
        """sql_within() builds a proper ST_Within(ST_Transform, ST_Transform) clause."""
        from sqlalchemy import Column, Integer
        from sqlalchemy.ext.declarative import declarative_base

        from geoalchemy2.types import Geometry

        from edisgo.io.db import sql_within

        Base = declarative_base()

        class GeoTable(Base):
            __tablename__ = "geo"
            id = Column(Integer, primary_key=True)
            geom = Column(Geometry("POLYGON", srid=4326))

        result = sql_within(GeoTable.geom, GeoTable.geom, 3035)

        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "ST_Within" in clause_str
        assert "ST_Transform" in clause_str

    def test_sql_intersects_returns_binary_expression(self):
        """sql_intersects() builds a proper ST_Intersects clause."""
        from sqlalchemy import Column, Integer
        from sqlalchemy.ext.declarative import declarative_base

        from geoalchemy2.types import Geometry

        from edisgo.io.db import sql_intersects

        Base = declarative_base()

        class GeoTable(Base):
            __tablename__ = "geo2"
            id = Column(Integer, primary_key=True)
            geom = Column(Geometry("POLYGON", srid=4326))

        result = sql_intersects(GeoTable.geom, GeoTable.geom, 3035)

        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "ST_Intersects" in clause_str
        assert "ST_Transform" in clause_str
