import logging
import os
import shutil

import pandas as pd
import pytest

from edisgo.network.dsm import DSM


class TestDSM:
    @pytest.yield_fixture(autouse=True)
    def setup_dsm_test_data(self):

        timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        self.p_max = pd.DataFrame(
            data={
                "load_1": [5.0, 6.0],
                "load_2": [7.0, 8.0],
            },
            index=timeindex,
        )
        self.p_min = pd.DataFrame(
            data={
                "load_1": [1.0, 2.0],
                "load_2": [3.0, 4.0],
            },
            index=timeindex,
        )
        self.e_max = pd.DataFrame(
            data={
                "load_1": [9.5, 10.5],
                "load_2": [0.9, 0.8],
            },
            index=timeindex,
        )
        self.e_min = pd.DataFrame(
            data={
                "load_1": [9.0, 10.0],
                "load_2": [0.7, 0.6],
            },
            index=timeindex,
        )
        self.dsm = DSM()
        self.dsm.p_max = self.p_max
        self.dsm.p_min = self.p_min
        self.dsm.e_max = self.e_max
        self.dsm.e_min = self.e_min

    def test_reduce_memory(self):

        # check with default value
        assert (self.dsm.p_max.dtypes == "float64").all()
        assert (self.dsm.e_max.dtypes == "float64").all()

        self.dsm.reduce_memory()

        assert (self.dsm.p_max.dtypes == "float32").all()
        assert (self.dsm.e_max.dtypes == "float32").all()

        # check arguments
        self.dsm.reduce_memory(to_type="float16", attr_to_reduce=["p_max"])

        assert (self.dsm.p_max.dtypes == "float16").all()
        assert (self.dsm.e_max.dtypes == "float32").all()

        # check with empty dataframes
        self.dsm.e_max = pd.DataFrame()
        self.dsm.reduce_memory()

    def test_to_csv(self):

        # test with default values
        save_dir = os.path.join(os.getcwd(), "dsm_csv")
        self.dsm.to_csv(save_dir)

        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 4
        assert "p_min.csv" in files_in_dir
        assert "p_max.csv" in files_in_dir
        assert "e_min.csv" in files_in_dir
        assert "e_max.csv" in files_in_dir

        shutil.rmtree(save_dir)

        # test with reduce memory True, to_type = float16
        self.dsm.to_csv(save_dir, reduce_memory=True, to_type="float16")

        assert (self.dsm.e_min.dtypes == "float16").all()
        files_in_dir = os.listdir(save_dir)
        assert len(files_in_dir) == 4

        shutil.rmtree(save_dir, ignore_errors=True)

    def test_from_csv(self):

        # write to csv
        save_dir = os.path.join(os.getcwd(), "dsm_csv")
        self.dsm.to_csv(save_dir)

        # reset DSM object
        self.dsm = DSM()

        self.dsm.from_csv(save_dir)

        pd.testing.assert_frame_equal(
            self.dsm.p_min,
            self.p_min,
            check_freq=False,
        )
        pd.testing.assert_frame_equal(
            self.dsm.e_min,
            self.e_min,
            check_freq=False,
        )

        shutil.rmtree(save_dir)

    def test_check_integrity(self, caplog):
        timeindex = pd.date_range("1/1/2011 12:00", periods=2, freq="H")
        # create duplicate entries and loads that do not appear in each DSM dataframe
        self.dsm.p_max = pd.concat(
            [
                self.dsm.p_max,
                pd.DataFrame(
                    data={
                        "load_2": [5.0, 6.0],
                        "load_3": [7.0, 8.0],
                    },
                    index=timeindex,
                ),
            ],
            axis=1,
        )
        self.dsm.p_min = pd.concat(
            [
                self.dsm.p_min,
                pd.DataFrame(
                    data={
                        "load_2": [5.0, 6.0],
                        "load_3": [7.0, 8.0],
                    },
                    index=timeindex,
                ),
            ],
            axis=1,
        )

        with caplog.at_level(logging.WARNING):
            self.dsm.check_integrity()
        assert len(caplog.messages) == 5
        assert "DSM timeseries contain the following duplicates:" in caplog.text
        assert "DSM timeseries e_min is missing the following entries:" in caplog.text
        assert "DSM timeseries e_max is missing the following entries:" in caplog.text
        assert (
            "DSM timeseries p_min contains values larger than zero, which is not "
            "allowed." in caplog.text
        )
        assert (
            "DSM timeseries e_min contains values larger than zero, which is not "
            "allowed." in caplog.text
        )

        caplog.clear()
        # check for empty DSM class
        self.dsm = DSM()
        with caplog.at_level(logging.WARNING):
            self.dsm.check_integrity()
        assert len(caplog.text) == 0
