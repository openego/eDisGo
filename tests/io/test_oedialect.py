import pytest
import sqlalchemy as sa

from sqlalchemy import exc

import edisgo.io.oedialect  # noqa: F401  (registers the dialect)

from edisgo.io.oedialect.api import resolve_params
from edisgo.io.oedialect.dialect import OEDialect


@pytest.fixture(scope="module")
def engine():
    return sa.create_engine("postgresql+oedialect://:@openenergyplatform.org")


@pytest.fixture(scope="module")
def table():
    return sa.Table(
        "feedin",
        sa.MetaData(),
        sa.Column("w_id", sa.Integer, primary_key=True),
        sa.Column("carrier", sa.Text),
        sa.Column("power_class", sa.Integer),
        schema="supply",
    )


def compile_statement(engine, statement):
    """Compile a statement and fill its bind parameters in, as the cursor does."""
    compiled = statement.compile(dialect=engine.dialect)
    return resolve_params(dict(compiled.string), compiled.params)


class TestOEDialect:
    """
    Tests for the OEP dialect.

    The dialect compiles statements into the JSON documents the OEP API takes
    instead of into SQL. The documents asserted on here are the ones the
    oedialect package produced with SQLAlchemy 1.3, which eDisGo used before.

    """

    def test_dialect_is_registered(self, engine):
        assert isinstance(engine.dialect, OEDialect)
        assert "oedialect" in str(engine.url)

    def test_column(self, engine, table):
        document = compile_statement(engine, sa.select(table.c.w_id))

        assert document["command"] == "advanced/search"
        assert document["type"] == "select"
        assert document["fields"] == [
            {
                "type": "column",
                "column": "w_id",
                "is_literal": False,
                "table": "feedin",
                "schema": "supply",
            }
        ]
        assert document["from"] == [
            {"type": "table", "schema": "supply", "table": "feedin"}
        ]

    def test_filter(self, engine, table):
        document = compile_statement(
            engine,
            sa.select(table.c.w_id).where(
                sa.and_(table.c.carrier == "solar", table.c.power_class == 2)
            ),
        )

        where = document["where"]
        assert where["type"] == "operator"
        assert where["operator"] == " AND "
        # the values of the two comparisons are filled in for the operands
        assert [operand["operands"][1] for operand in where["operands"]] == ["solar", 2]

    def test_in_clause(self, engine, table):
        # SQLAlchemy renders an IN clause as one parameter that it substitutes
        # into the SQL string; the dialect has to put the values into the
        # document instead
        document = compile_statement(
            engine, sa.select(table.c.w_id).where(table.c.w_id.in_([1, 2]))
        )

        assert document["where"]["operator"] == " IN "
        assert document["where"]["operands"][1] == {
            "type": "grouping",
            "grouping": [1, 2],
        }

    def test_order_by_and_limit(self, engine, table):
        document = compile_statement(
            engine, sa.select(table.c.carrier).order_by(table.c.w_id).limit(5)
        )

        assert document["limit"] == 5
        assert [column["column"] for column in document["order_by"]] == ["w_id"]

    def test_write_statements_are_rejected(self, engine, table):
        # eDisGo only reads from the OEP, so the dialect does not compile
        # statements that change data
        for statement in [
            table.insert().values(w_id=1),
            table.update().values(carrier="solar"),
            table.delete(),
        ]:
            with pytest.raises(exc.CompileError, match="read-only"):
                statement.compile(dialect=engine.dialect)


class TestResolveParams:
    def test_bind_parameters_are_called(self):
        document = {"limit": lambda params: params["limit_1"]}

        assert resolve_params(document, {"limit_1": 3}) == {"limit": 3}

    def test_none_becomes_a_null_value(self):
        assert resolve_params({"value": None}, {}) == {
            "value": {"type": "value", "value": None}
        }

    def test_nested_documents(self):
        document = {"operands": [{"a": lambda params: params["x"]}, [1, "b"]]}

        assert resolve_params(document, {"x": 7}) == {"operands": [{"a": 7}, [1, "b"]]}
