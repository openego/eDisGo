"""Tests for edisgo.io.mviews_filters — scenario filter builders."""

from sqlalchemy import Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base

from edisgo.io.mviews_filters import (
    build_conv_scenario_filter,
    build_res_scenario_filter,
)

Base = declarative_base()


class MockConvTable(Base):
    """Mock table mimicking ego_dp_conv_powerplant structure."""

    __tablename__ = "mock_conv"
    id = Column(Integer, primary_key=True)
    capacity = Column(Float)
    preversion = Column(String)
    version = Column(String)
    scenario = Column(String)
    fuel = Column(String)
    shutdown = Column(Integer)


class MockResTable(Base):
    """Mock table mimicking ego_dp_res_powerplant structure."""

    __tablename__ = "mock_res"
    id = Column(Integer, primary_key=True)
    electrical_capacity = Column(Float)
    preversion = Column(String)
    version = Column(String)
    scenario = Column(String)
    generation_type = Column(String)
    generation_subtype = Column(String)


class TestBuildConvScenarioFilter:
    def test_nep2035_default_versions(self):
        result = build_conv_scenario_filter(MockConvTable, "NEP 2035")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "NEP 2035" in clause_str
        assert "v0.4.2" in clause_str

    def test_ego100_default_versions(self):
        result = build_conv_scenario_filter(MockConvTable, "eGo 100")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "pumped_storage" in clause_str

    def test_generic_scenario(self):
        result = build_conv_scenario_filter(MockConvTable, "Status Quo")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "Status Quo" in clause_str

    def test_custom_version_string(self):
        result = build_conv_scenario_filter(MockConvTable, "NEP 2035", version="v1.0")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "v1.0" in clause_str

    def test_custom_version_list(self):
        result = build_conv_scenario_filter(
            MockConvTable, "NEP 2035", version=["v1.0", "v2.0"]
        )
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "v1.0" in clause_str
        assert "v2.0" in clause_str


class TestBuildResScenarioFilter:
    def test_status_quo(self):
        result = build_res_scenario_filter(MockResTable, "Status Quo")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "Status Quo" in clause_str
        assert "solar" in clause_str

    def test_nep2035_union(self):
        result = build_res_scenario_filter(MockResTable, "NEP 2035")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        # NEP 2035 is a UNION of Status Quo + NEP 2035
        assert "Status Quo" in clause_str
        assert "NEP 2035" in clause_str

    def test_generic_scenario(self):
        result = build_res_scenario_filter(MockResTable, "eGo 100")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "eGo 100" in clause_str

    def test_custom_version(self):
        result = build_res_scenario_filter(MockResTable, "Status Quo", version="v1.0")
        clause_str = str(result.compile(compile_kwargs={"literal_binds": True}))
        assert "v1.0" in clause_str
