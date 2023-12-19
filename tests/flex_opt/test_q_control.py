import numpy as np
import pandas as pd

from edisgo.flex_opt import q_control
from edisgo.tools.config import Config


class TestQControl:
    def test_get_q_sign_generator(self):
        assert q_control.get_q_sign_generator("Inductive") == -1
        assert q_control.get_q_sign_generator("capacitive") == 1

    def test_get_q_sign_load(self):
        assert q_control.get_q_sign_load("inductive") == 1
        assert q_control.get_q_sign_load("Capacitive") == -1

    def test_fixed_cosphi(self):
        timeindex = pd.date_range("1/1/1970", periods=2, freq="H")
        active_power_ts = pd.DataFrame(
            data={
                "comp_mv_1": [0.5, 1.5],
                "comp_mv_2": [2.5, 3.5],
                "comp_lv_1": [0.1, 0.0],
                "comp_lv_2": [0.15, 0.07],
            },
            index=timeindex,
        )
        q_sign = pd.Series(
            [-1.0, 1.0, 1.0, -1],
            index=["comp_mv_1", "comp_mv_2", "comp_lv_1", "comp_lv_2"],
        )
        power_factor = pd.Series(
            [0.9, 0.95, 1.0, 0.9],
            index=["comp_mv_1", "comp_mv_2", "comp_lv_1", "comp_lv_2"],
        )

        # test with q_sign as Series and power_factor as float
        reactive_power_ts = q_control.fixed_cosphi(
            active_power_ts,
            q_sign=q_sign,
            power_factor=0.9,
        )

        assert reactive_power_ts.shape == (2, 4)
        assert np.isclose(
            reactive_power_ts.loc[:, ["comp_mv_1", "comp_lv_2"]].values,
            active_power_ts.loc[:, ["comp_mv_1", "comp_lv_2"]].values * -0.484322,
        ).all()
        assert np.isclose(
            reactive_power_ts.loc[:, "comp_lv_1"].values,
            active_power_ts.loc[:, "comp_lv_1"].values * 0.484322,
        ).all()

        # test with q_sign as int and power_factor as Series
        reactive_power_ts = q_control.fixed_cosphi(
            active_power_ts,
            q_sign=1,
            power_factor=power_factor,
        )

        assert reactive_power_ts.shape == (2, 4)
        assert np.isclose(
            reactive_power_ts.loc[:, ["comp_mv_1", "comp_lv_2"]].values,
            active_power_ts.loc[:, ["comp_mv_1", "comp_lv_2"]].values * 0.484322,
        ).all()
        assert np.isclose(
            reactive_power_ts.loc[:, "comp_lv_1"].values,
            [0.0, 0.0],
        ).all()
        assert np.isclose(
            reactive_power_ts.loc[:, "comp_mv_2"].values,
            active_power_ts.loc[:, "comp_mv_2"].values * 0.328684,
        ).all()

        # test with q_sign as int and power_factor as float
        reactive_power_ts = q_control.fixed_cosphi(
            active_power_ts,
            q_sign=1,
            power_factor=0.95,
        )

        assert reactive_power_ts.shape == (2, 4)
        assert np.isclose(
            reactive_power_ts.loc[
                :, ["comp_mv_1", "comp_mv_2", "comp_lv_1", "comp_lv_2"]
            ].values,
            active_power_ts.loc[
                :, ["comp_mv_1", "comp_mv_2", "comp_lv_1", "comp_lv_2"]
            ].values
            * 0.328684,
        ).all()

    def test__fixed_cosphi_default_power_factor(
        self,
    ):
        df = pd.DataFrame(
            data={"voltage_level": ["mv", "lv", "lv"]},
            index=["comp_mv_1", "comp_lv_1", "comp_lv_2"],
        )
        config = Config()

        # test for component_type="generators"
        pf = q_control._fixed_cosphi_default_power_factor(
            comp_df=df, component_type="generators", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [0.9, 0.95, 0.95],
        ).all()

        # test for component_type="loads"
        pf = q_control._fixed_cosphi_default_power_factor(
            comp_df=df, component_type="conventional_loads", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [0.9, 0.95, 0.95],
        ).all()

        # test for component_type="charging_points"
        pf = q_control._fixed_cosphi_default_power_factor(
            comp_df=df, component_type="charging_points", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [1.0, 1.0, 1.0],
        ).all()

        # test for component_type="heat_pumps"
        pf = q_control._fixed_cosphi_default_power_factor(
            comp_df=df, component_type="heat_pumps", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [1.0, 1.0, 1.0],
        ).all()

        # test for component_type="storage_units"
        pf = q_control._fixed_cosphi_default_power_factor(
            comp_df=df, component_type="storage_units", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [0.9, 0.95, 0.95],
        ).all()

    def test__fixed_cosphi_default_reactive_power_sign(
        self,
    ):
        df = pd.DataFrame(
            data={"voltage_level": ["mv", "lv", "lv"]},
            index=["comp_mv_1", "comp_lv_1", "comp_lv_2"],
        )
        config = Config()

        # test for component_type="generators"
        pf = q_control._fixed_cosphi_default_reactive_power_sign(
            comp_df=df, component_type="generators", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [-1.0, -1.0, -1.0],
        ).all()

        # test for component_type="conventional_loads"
        pf = q_control._fixed_cosphi_default_reactive_power_sign(
            comp_df=df, component_type="conventional_loads", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [1.0, 1.0, 1.0],
        ).all()

        # test for component_type="charging_points"
        pf = q_control._fixed_cosphi_default_reactive_power_sign(
            comp_df=df, component_type="charging_points", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [1.0, 1.0, 1.0],
        ).all()

        # test for component_type="heat_pumps"
        pf = q_control._fixed_cosphi_default_reactive_power_sign(
            comp_df=df, component_type="heat_pumps", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [1.0, 1.0, 1.0],
        ).all()

        # test for component_type="storage_units"
        pf = q_control._fixed_cosphi_default_reactive_power_sign(
            comp_df=df, component_type="storage_units", configs=config
        )

        assert pf.shape == (3,)
        assert np.isclose(
            pf.loc[["comp_mv_1", "comp_lv_1", "comp_lv_2"]].values,
            [-1.0, -1.0, -1.0],
        ).all()
