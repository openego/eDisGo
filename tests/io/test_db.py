"""
Tests for the database source selection in :mod:`edisgo.io.db`.

These tests only exercise the routing logic (which database is chosen and
how the connection URL is built) — no actual database or SSH connection is
established. The OEP path is covered by asserting on the returned engine's
URL; the egon-data paths are covered with temporary configuration files and
a monkeypatched ``ssh_tunnel``.
"""
import textwrap

import pytest

from edisgo.io import db


def _write_config(path, ssh_tunnel_section=False):
    config = textwrap.dedent(
        """\
        egon-data:
          --database-name: egon-data
          --database-host: 127.0.0.1
          --database-port: 58888
          --database-user: egon
          --database-password: data
        """
    )
    if ssh_tunnel_section:
        config += textwrap.dedent(
            """\
            ssh-tunnel:
              ssh-host: gateway.example.org
              ssh-user: tunneluser
              ssh-pkey: ~/.ssh/id_ed25519
              pgres-host: 127.0.0.1
            """
        )
    path.write_text(config)
    return path


class TestDefaultConfigPath:
    def test_env_variable_overrides_default(self, tmp_path, monkeypatch):
        config = _write_config(tmp_path / "egon-data.configuration.yaml")
        monkeypatch.setenv("EGON_DATA_CONFIG", str(config))
        assert db.default_config_path() == config

    def test_returns_none_if_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EGON_DATA_CONFIG", str(tmp_path / "missing.yaml"))
        assert db.default_config_path() is None


class TestEngine:
    def test_explicit_oep(self):
        engine = db.engine(ssh=False)
        assert "oedialect" in str(engine.url)

    def test_auto_detect_falls_back_to_oep(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EGON_DATA_CONFIG", str(tmp_path / "missing.yaml"))
        engine = db.engine()
        assert "oedialect" in str(engine.url)

    def test_auto_detect_uses_egon_data_config(self, tmp_path, monkeypatch):
        config = _write_config(tmp_path / "egon-data.configuration.yaml")
        monkeypatch.setenv("EGON_DATA_CONFIG", str(config))
        engine = db.engine()
        assert engine.url.database == "egon-data"
        assert engine.url.port == 58888

    def test_explicit_path_without_tunnel_connects_directly(self, tmp_path):
        config = _write_config(tmp_path / "egon-data.configuration.yaml")
        engine = db.engine(path=config)
        assert engine.url.host == "127.0.0.1"
        assert engine.url.port == 58888
        assert engine.url.username == "egon"

    def test_config_with_tunnel_section_opens_tunnel(
        self, tmp_path, monkeypatch
    ):
        config = _write_config(
            tmp_path / "egon-data.configuration.yaml", ssh_tunnel_section=True
        )
        # The private key check in credentials() must find a file.
        pkey = tmp_path / "id_ed25519"
        pkey.write_text("dummy")
        text = config.read_text().replace("~/.ssh/id_ed25519", str(pkey))
        config.write_text(text)

        opened = {}

        def fake_tunnel(cred):
            opened["cred"] = cred
            return "55555"

        monkeypatch.setattr(db, "ssh_tunnel", fake_tunnel)
        engine = db.engine(path=config)
        assert opened["cred"]["SSH_HOST"] == "gateway.example.org"
        assert engine.url.port == 55555

    def test_ssh_true_without_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EGON_DATA_CONFIG", str(tmp_path / "missing.yaml"))
        with pytest.raises(ValueError, match="no configuration file"):
            db.engine(ssh=True)

    def test_explicit_missing_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            db.engine(path=tmp_path / "missing.yaml")


class TestEngineFromSettings:
    def test_source_oep(self):
        engine = db.engine_from_settings({"source": "oep"})
        assert "oedialect" in str(engine.url)

    def test_source_egon_data_with_config_path(self, tmp_path):
        config = _write_config(tmp_path / "egon-data.configuration.yaml")
        engine = db.engine_from_settings(
            {"source": "egon-data", "config_path": str(config)}
        )
        assert engine.url.database == "egon-data"

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown database source"):
            db.engine_from_settings({"source": "local"})

    def test_empty_settings_auto_detect(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EGON_DATA_CONFIG", str(tmp_path / "missing.yaml"))
        engine = db.engine_from_settings(None)
        assert "oedialect" in str(engine.url)
