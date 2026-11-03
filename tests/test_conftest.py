import pytest
import sqlalchemy as sa

# The guard rejects these connections before they are opened, so the tests never
# reach the OEP. The timeout only bounds the test if the guard is missing.
OEP_URLS = [
    "postgresql+oedialect://:token@openenergyplatform.org",
    "postgresql+psycopg2://user:password@openenergyplatform.org/oedb?connect_timeout=5",
]


@pytest.mark.parametrize("url", OEP_URLS, ids=["oedialect", "postgres-host"])
@pytest.mark.parametrize(
    "connect",
    [
        lambda engine: engine.connect(),
        lambda engine: engine.raw_connection(),
        # table reflection connects without Engine.connect in SQLAlchemy 1.3
        lambda engine: sa.Table(
            "cosmoclmgrid", sa.MetaData(bind=engine), schema="climate", autoload=True
        ),
    ],
    ids=["connect", "raw-connection", "reflection"],
)
def test_unmarked_test_cannot_connect_to_oep(url, connect):
    engine = sa.create_engine(url, poolclass=sa.pool.NullPool)

    with pytest.raises(pytest.fail.Exception, match="Unmarked test attempted"):
        connect(engine)
