import os
import socket

import pytest

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
