import pytest

from edisgo import EDisGo
from edisgo.tools.plots import dash_plot


class TestPlots:
    @classmethod
    def setup_class(cls):
        cls.edisgo = EDisGo(ding0_grid=pytest.ding0_test_network_path)
        cls.edisgo.set_time_series_worst_case_analysis()
        cls.edisgo.reinforce()

    def test_dash_plot(self):
        # TODO: at the moment this doesn't really test anything. Add meaningful tests.
        # test if any errors occur when only passing one edisgo object
        app = dash_plot(
            edisgo_objects=self.edisgo,
        )

        # test if any errors occur when passing multiple edisgo objects
        app = dash_plot(  # noqa: F841
            edisgo_objects={
                "edisgo_1": self.edisgo,
                "edisgo_2": self.edisgo,
            }
        )
