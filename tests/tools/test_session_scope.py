import importlib

from sqlalchemy import create_engine, text

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


def test_legacy_oep_table_reflects_table_from_oep_engine(monkeypatch):
    # the legacy tables are reflected with saio from the engine of
    # session_scope, so no ORM definitions (formerly egoio) are needed
    oep_engine = create_engine("sqlite://")
    with oep_engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE ego_renewable_feedin "
                "(w_id INTEGER PRIMARY KEY, source TEXT)"
            )
        )

    monkeypatch.setattr(edisgo.io.db, "engine", lambda **kwargs: oep_engine)
    tools._oep_engine.cache_clear()
    try:
        table = tools.legacy_oep_table("main", "ego_renewable_feedin")
    finally:
        tools._oep_engine.cache_clear()

    assert table.__table__.schema == "main"
    assert set(table.__table__.columns.keys()) == {"w_id", "source"}
    assert importlib.import_module("saio.main").engine is oep_engine
