import os
import shutil

import numpy as np
import pandas as pd
import pytest

from edisgo import EDisGo
from edisgo.network.overlying_grid import (
    OverlyingGrid,
    distribute_overlying_grid_requirements,
)


class TestOverlyingGrid:
    """
    Tests OverlyingGrid class.

    """

    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        self.overlying_grid = OverlyingGrid()
        self.timeindex = pd.date_range("1/1/2018", periods=2, freq="H")
        self.overlying_grid.renewables_curtailment = pd.Series(
            data=[2.4], index=[self.timeindex[0]]
        )
        self.overlying_grid.feedin_district_heating = pd.DataFrame(
            {"dh1": [1.4, 2.3], "dh2": [2.4, 1.3]}, index=self.timeindex
        )

    def test_reduce_memory(self):
        # check with default value
        assert self.overlying_grid.renewables_curtailment.dtypes == "float64"
        assert (self.overlying_grid.feedin_district_heating.dtypes == "float64").all()

        self.overlying_grid.reduce_memory()
        assert self.overlying_grid.renewables_curtailment.dtypes == "float32"
        assert (self.overlying_grid.feedin_district_heating.dtypes == "float32").all()

        # check with arguments
        self.overlying_grid.reduce_memory(
            to_type="float16",
            attr_to_reduce=["renewables_curtailment"],
        )

        assert self.overlying_grid.renewables_curtailment.dtypes == "float16"
        assert (self.overlying_grid.feedin_district_heating.dtypes == "float32").all()

    def test_to_csv(self):
        # test with default values
        save_dir = os.path.join(os.getcwd(), "overlying_grid_csv")
        self.overlying_grid.to_csv(save_dir)

        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 2
        assert "renewables_curtailment.csv" in files_in_dir
        assert "feedin_district_heating.csv" in files_in_dir

        shutil.rmtree(save_dir)

        # test with reduce memory True and to_type = float16
        self.overlying_grid.to_csv(save_dir, reduce_memory=True, to_type="float16")

        assert (self.overlying_grid.feedin_district_heating.dtypes == "float16").all()
        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 2

        shutil.rmtree(save_dir, ignore_errors=True)

    def test_from_csv(self):
        renewables_curtailment = self.overlying_grid.renewables_curtailment
        feedin_district_heating = self.overlying_grid.feedin_district_heating

        # write to csv
        save_dir = os.path.join(os.getcwd(), "overlying_grid_csv")
        self.overlying_grid.to_csv(save_dir)

        # reset OverlyingGrid
        self.overlying_grid = OverlyingGrid()

        # test with default parameters
        self.overlying_grid.from_csv(save_dir)

        pd.testing.assert_series_equal(
            self.overlying_grid.renewables_curtailment,
            renewables_curtailment,
            check_names=False,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            self.overlying_grid.feedin_district_heating,
            feedin_district_heating,
            check_freq=False,
        )

        # test with dtype = float32
        self.overlying_grid.from_csv(save_dir, dtype="float32")
        assert (self.overlying_grid.feedin_district_heating.dtypes == "float32").all()

        shutil.rmtree(save_dir)

    def test_resample(self, caplog):
        mean_value_curtailment_orig = self.overlying_grid.renewables_curtailment.mean()
        mean_value_feedin_dh_orig = self.overlying_grid.feedin_district_heating.mean()

        # test up-sampling with ffill (default)
        self.overlying_grid.resample()
        # check if resampled length of time index is 4 times original length of
        # timeindex
        assert len(self.overlying_grid.feedin_district_heating.index) == 4 * len(
            self.timeindex
        )
        # check if mean value of resampled data is the same as mean value of original
        # data
        assert np.isclose(
            self.overlying_grid.renewables_curtailment.mean(),
            mean_value_curtailment_orig,
            atol=1e-5,
        )
        assert (
            np.isclose(
                self.overlying_grid.feedin_district_heating.mean(),
                mean_value_feedin_dh_orig,
                atol=1e-5,
            )
        ).all()
        # check if index is the same after resampled back
        self.overlying_grid.resample(freq="1h")
        pd.testing.assert_index_equal(
            self.overlying_grid.feedin_district_heating.index,
            self.timeindex,
        )

        # same tests for down-sampling
        self.overlying_grid.resample(freq="2h")
        assert len(self.overlying_grid.feedin_district_heating.index) == 0.5 * len(
            self.timeindex
        )
        assert np.isclose(
            self.overlying_grid.renewables_curtailment.mean(),
            mean_value_curtailment_orig,
            atol=1e-5,
        )
        assert (
            np.isclose(
                self.overlying_grid.feedin_district_heating.mean(),
                mean_value_feedin_dh_orig,
                atol=1e-5,
            )
        ).all()

        # test warning that resampling cannot be conducted
        self.overlying_grid.resample()
        assert (
            "Data cannot be resampled as it only contains one time step." in caplog.text
        )


class TestOverlyingGridFunc:
    @classmethod
    def setup_class(self):
        self.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        self.edisgo.set_time_series_worst_case_analysis()
        self.timesteps = self.edisgo.timeseries.timeindex

    def setup_flexibility_data(self):
        # add heat pump dummy data
        self.edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            sector="individual_heating",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex,
                data=[1.0 / 5, 2.0 / 6, 2.0 / 5, 1.0 / 6],
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[26],
            p_set=2,
        )
        self.edisgo.add_component(
            comp_type="load",
            type="heat_pump",
            sector="individual_heating",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex,
                data=[2.0 / 7.0, 4.0 / 8.0, 3.0 / 7.0, 3.0 / 8.0],
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[30],
            p_set=3,
        )

        # add electromobility dummy data
        self.edisgo.add_component(
            comp_type="load",
            type="charging_point",
            ts_active_power=pd.Series(
                index=self.edisgo.timeseries.timeindex, data=[0.5, 0.5, 0.5, 0.5]
            ),
            ts_reactive_power="default",
            bus=self.edisgo.topology.buses_df.index[32],
            p_set=3,
        )
        flex_bands = {
            "lower_energy": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [0, 0, 1, 2]},
                index=self.edisgo.timeseries.timeindex,
            ),
            "upper_energy": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [1, 2, 2, 3]},
                index=self.edisgo.timeseries.timeindex,
            ),
            "upper_power": pd.DataFrame(
                {"Charging_Point_LVGrid_6_1": [1, 1, 2, 1]},
                index=self.edisgo.timeseries.timeindex,
            ),
        }
        self.edisgo.electromobility.flexibility_bands = flex_bands

        # add DSM dummy data
        self.edisgo.dsm.p_min = pd.DataFrame(
            data={
                "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
                    -0.3,
                    -0.3,
                    -0.3,
                    -0.3,
                ],
                "Load_industrial_LVGrid_5_1": [-0.07, -0.07, -0.07, -0.07],
            },
            index=self.edisgo.timeseries.timeindex,
        )
        self.edisgo.dsm.p_max = pd.DataFrame(
            data={
                "Load_retail_MVGrid_1_Load_aggregated_retail_MVGrid_1_1": [
                    0.3,
                    0.3,
                    0.3,
                    0.3,
                ],
                "Load_industrial_LVGrid_5_1": [0.07, 0.07, 0.07, 0.07],
            },
            index=self.edisgo.timeseries.timeindex,
        )

        # add overlying grid dummy data
        for attr in [
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "renewables_curtailment",
            "storage_units_active_power",
        ]:
            if attr == "dsm_active_power":
                data = [0.1, -0.1, -0.1, 0.1]
            elif attr == "electromobility_active_power":
                data = [0.4, 0.5, 0.5, 0.6]
            elif attr == "heat_pump_decentral_active_power":
                data = [0.5, 0.85, 0.85, 0.55]
            elif attr == "storage_units_active_power":
                data = [-0.35, -0.35, 0.35, 0.35]
            elif attr == "renewables_curtailment":
                data = [0, 0, 0.1, 0.1]

            df = pd.Series(
                index=self.timesteps,
                data=data,
            )
            setattr(
                self.edisgo.overlying_grid,
                attr,
                df,
            )

        # Resample timeseries and reindex to hourly timedelta
        self.edisgo.resample_timeseries(freq="1min")

        for attr in ["p_min", "p_max"]:
            new_dates = pd.DatetimeIndex(
                [getattr(self.edisgo.dsm, attr).index[-1] + pd.Timedelta("1h")]
            )
            setattr(
                self.edisgo.dsm,
                attr,
                getattr(self.edisgo.dsm, attr)
                .reindex(
                    getattr(self.edisgo.dsm, attr)
                    .index.union(new_dates)
                    .unique()
                    .sort_values()
                )
                .ffill()
                .resample("1min")
                .ffill()
                .iloc[:-1],
            )
        self.timesteps = pd.date_range(start="01/01/2018", periods=240, freq="h")
        attributes = self.edisgo.timeseries._attributes
        for attr in attributes:
            if not getattr(self.edisgo.timeseries, attr).empty:
                df = pd.DataFrame(
                    index=self.timesteps,
                    columns=getattr(self.edisgo.timeseries, attr).columns,
                    data=getattr(self.edisgo.timeseries, attr).values,
                )
                setattr(
                    self.edisgo.timeseries,
                    attr,
                    df,
                )
        self.edisgo.timeseries.timeindex = self.timesteps
        # Battery electric vehicle timeseries
        for key, df in self.edisgo.electromobility.flexibility_bands.items():
            if not df.empty:
                df.index = self.timesteps
                self.edisgo.electromobility.flexibility_bands.update({key: df})
        # Demand Side Management timeseries
        for attr in ["p_min", "p_max"]:
            if not getattr(self.edisgo.dsm, attr).empty:
                df = pd.DataFrame(
                    index=self.timesteps,
                    columns=getattr(self.edisgo.dsm, attr).columns,
                    data=getattr(self.edisgo.dsm, attr).values,
                )
                setattr(
                    self.edisgo.dsm,
                    attr,
                    df,
                )
        # overlying grid timeseries
        for attr in [
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "renewables_curtailment",
            "storage_units_active_power",
        ]:
            if not getattr(self.edisgo.overlying_grid, attr).empty:
                df = pd.Series(
                    index=self.timesteps,
                    data=getattr(self.edisgo.overlying_grid, attr).values,
                )
                setattr(
                    self.edisgo.overlying_grid,
                    attr,
                    df,
                )

    def test_distribute_overlying_grid_timeseries(self):
        self.setup_flexibility_data()
        edisgo_copy = distribute_overlying_grid_requirements(self.edisgo)

        hps = self.edisgo.topology.loads_df.index[
            self.edisgo.topology.loads_df.sector.isin(
                ["individual_heating", "individual_heating_resistive_heater"]
            )
        ]

        assert np.isclose(
            edisgo_copy.timeseries.loads_active_power[hps].sum(axis=1),
            self.edisgo.overlying_grid.heat_pump_decentral_active_power,
            atol=1e-5,
        ).all()
        assert (
            edisgo_copy.timeseries.loads_active_power["Charging_Point_LVGrid_6_1"]
            == self.edisgo.overlying_grid.electromobility_active_power.values
        ).all()
        assert (
            edisgo_copy.timeseries.storage_units_active_power["Storage_1"]
            == self.edisgo.overlying_grid.storage_units_active_power.values
        ).all()

        dsm = self.edisgo.dsm.p_max.columns.values
        assert np.isclose(
            edisgo_copy.timeseries.loads_active_power[dsm].sum(axis=1),
            self.edisgo.timeseries.loads_active_power[dsm].sum(axis=1)
            + self.edisgo.overlying_grid.dsm_active_power,
            atol=1e-5,
        ).all()

        res = self.edisgo.topology.generators_df.loc[
            (self.edisgo.topology.generators_df.type == "solar")
            | (self.edisgo.topology.generators_df.type == "wind")
        ].index.values
        assert np.isclose(
            edisgo_copy.timeseries.generators_active_power[res].sum(axis=1),
            self.edisgo.timeseries.generators_active_power[res].sum(axis=1)
            - self.edisgo.overlying_grid.renewables_curtailment,
            atol=1e-5,
        ).all()
