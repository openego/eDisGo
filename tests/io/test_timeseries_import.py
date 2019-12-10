import pandas as pd
import pytest

from edisgo.io.timeseries_import import import_feedin_timeseries
from edisgo.tools.config import Config


class TestTimeseriesImport:

    def test_import_feedin_timeseries(self):
        config = Config(config_path=None)
        weather_cells = [1122074., 1122075.]
        timeindex = pd.date_range('1/1/2011', periods=8760, freq='H')
        feedin = import_feedin_timeseries(config, weather_cells, timeindex)
        assert len(feedin['solar'][1122074]) == 8760
        assert len(feedin['solar'][1122075]) == 8760
        assert len(feedin['wind'][1122074]) == 8760
        assert len(feedin['wind'][1122075]) == 8760
        assert feedin['solar'][1122074][timeindex[13]] == 0.074941092034683
        assert feedin['wind'][1122074][timeindex[37]] == 0.039784172908844
        assert feedin['solar'][1122075][timeindex[61]] == 0.423822557381157
        assert feedin['wind'][1122075][timeindex[1356]] == 0.10636113747161

        # check trying to import different year
        msg = "The year you inserted could not be imported from " \
              "the oedb. So far only 2011 is provided. Please " \
              "check website for updates."
        timeindex = pd.date_range('1/1/2018', periods=8760, freq='H')
        with pytest.raises(ValueError, match=msg):
            feedin = import_feedin_timeseries(config, weather_cells, timeindex)
