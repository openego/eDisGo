import numpy as np
import pandas as pd
import pytest

from edisgo.io import timeseries_import
from edisgo.tools.config import Config


class TestTimeseriesImport:

    @classmethod
    def setup_class(self):
        self.config = Config(config_path=None)

    def test_feedin_oedb(self):
        weather_cells = [1122074., 1122075.]
        timeindex = pd.date_range('1/1/2011', periods=8760, freq='H')
        feedin = timeseries_import.feedin_oedb(
            self.config, weather_cells, timeindex)
        assert len(feedin["solar"][1122074]) == 8760
        assert len(feedin["solar"][1122075]) == 8760
        assert len(feedin["wind"][1122074]) == 8760
        assert len(feedin["wind"][1122075]) == 8760
        assert np.isclose(
            feedin["solar"][1122074][timeindex[13]], 0.074941)
        assert np.isclose(
            feedin["wind"][1122074][timeindex[37]], 0.039784)
        assert np.isclose(
            feedin["solar"][1122075][timeindex[61]], 0.423823)
        assert np.isclose(
            feedin["wind"][1122075][timeindex[1356]], 0.106361)

        # check trying to import different year
        msg = (
            "The year you inserted could not be imported from "
            "the oedb. So far only 2011 is provided. Please "
            "check website for updates."
        )
        timeindex = pd.date_range("1/1/2018", periods=8760, freq="H")
        with pytest.raises(ValueError, match=msg):
            feedin = timeseries_import.feedin_oedb(
                self.config, weather_cells, timeindex)

    def test_import_load_timeseries(self):
        timeindex = pd.date_range("1/1/2018", periods=8760, freq="H")
        load = timeseries_import.load_time_series_demandlib(
            self.config, timeindex[0].year)
        assert (load.columns == ["retail", "residential",
                                 "agricultural", "industrial"]).all()
        assert np.isclose(load.loc[timeindex[453], "retail"],
                          8.33507e-05)
        assert np.isclose(load.loc[timeindex[13], "residential"],
                          1.73151e-04)
        assert np.isclose(load.loc[timeindex[6328], "agricultural"],
                          1.01346e-04)
        assert np.isclose(load.loc[timeindex[4325], "industrial"],
                          9.91768e-05)
