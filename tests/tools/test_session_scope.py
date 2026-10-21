from sqlalchemy import create_engine

import edisgo.io.db

from edisgo import tools


def test_session_scope_uses_oep_engine(monkeypatch):
    # the legacy session must use edisgo's OEP engine (which reads the OEP
    # token) instead of an anonymous connection, force the OEP instead of
    # auto-detecting a local egon-data database, and reuse one engine
    calls = []
    oep_engine = create_engine("sqlite://")

    def fake_engine(**kwargs):
        calls.append(kwargs)
        return oep_engine

    monkeypatch.setattr(edisgo.io.db, "engine", fake_engine)
    tools._oep_engine.cache_clear()
    try:
        with tools.session_scope() as session:
            assert session.bind is oep_engine
        with tools.session_scope():
            pass
    finally:
        tools._oep_engine.cache_clear()

    assert calls == [{"ssh": False}]
