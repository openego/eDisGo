import os
import shutil

import numpy as np
import pandas as pd
import pytest

from edisgo.network.overlying_grid import OverlyingGrid


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
