from __future__ import annotations

import logging
import os

from copy import deepcopy
from zipfile import ZipFile

import pandas as pd

# from edisgo import EDisGo
from edisgo.tools.tools import resample

logger = logging.getLogger(__name__)


class OverlyingGrid:
    """
    Data container for requirements from the overlying grid.

    The requirements from the overlying grid are used as constraints for flexibilities.

    Attributes
    -----------
    renewables_curtailment : :pandas:`pandas.Series<Series>`
        Curtailment of fluctuating renewables per time step in MW.
    storage_units_active_power : :pandas:`pandas.Series<Series>`
        Aggregated dispatch of storage units per time step in MW.
    storage_units_soc : :pandas:`pandas.Series<Series>`
        State of charge of storage units per time step in p.u.. The state of charge at
        time step t here constitutes the state of charge at the beginning of time step
        t.
    dsm_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand side management utilisation per time step in MW.
    electromobility_active_power : :pandas:`pandas.Series<Series>`
        Aggregated charging demand at all charging sites in grid per time step in MW.
    heat_pump_decentral_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand of flexible decentral heat pumps per time step in MW.
    thermal_storage_units_decentral_soc : :pandas:`pandas.Series<Series>`
        State of charge of decentral thermal storage units in p.u..
    heat_pump_central_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand of flexible central heat pumps per time step in MW.
    thermal_storage_units_central_soc : :pandas:`pandas.DataFrame<DataFrame>`
        State of charge of central thermal storage units per district heating area (in
        columns as string of integer, i.e. "130" instead of "130.0") and time step
        (in index) in p.u.. The state of charge at time step t here constitutes the
        state of charge at the beginning of time step t.
    feedin_district_heating : :pandas:`pandas.DataFrame<DataFrame>`
        Other thermal feed-in into district heating per district heating area (in
        columns as string of integer, i.e. "130" instead of "130.0") and time step
        (in index) in MW.

    """

    def __init__(self, **kwargs):
        self.renewables_curtailment = kwargs.get(
            "renewables_curtailment", pd.Series(dtype="float64")
        )
        self.storage_units_active_power = kwargs.get(
            "storage_units_active_power", pd.Series(dtype="float64")
        )
        self.storage_units_soc = kwargs.get(
            "storage_units_soc", pd.Series(dtype="float64")
        )
        self.dsm_active_power = kwargs.get(
            "dsm_active_power", pd.Series(dtype="float64")
        )
        self.electromobility_active_power = kwargs.get(
            "electromobility_active_power", pd.Series(dtype="float64")
        )
        self.heat_pump_decentral_active_power = kwargs.get(
            "heat_pump_decentral_active_power", pd.Series(dtype="float64")
        )
        self.thermal_storage_units_decentral_soc = kwargs.get(
            "thermal_storage_units_decentral_soc", pd.Series(dtype="float64")
        )
        self.heat_pump_central_active_power = kwargs.get(
            "heat_pump_central_active_power", pd.Series(dtype="float64")
        )
        self.thermal_storage_units_central_soc = kwargs.get(
            "thermal_storage_units_central_soc", pd.DataFrame(dtype="float64")
        )
        self.feedin_district_heating = kwargs.get(
            "feedin_district_heating", pd.DataFrame(dtype="float64")
        )

    @property
    def _attributes(self):
        return [
            "renewables_curtailment",
            "storage_units_active_power",
            "storage_units_soc",
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "thermal_storage_units_decentral_soc",
            "heat_pump_central_active_power",
            "thermal_storage_units_central_soc",
            "feedin_district_heating",
        ]

    def reduce_memory(self, attr_to_reduce=None, to_type="float32"):
        """
        Reduces size of time series data to save memory.

        Parameters
        -----------
        attr_to_reduce : list(str), optional
            List of attributes to reduce size for. Per default, all time series data
            are reduced.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        if attr_to_reduce is None:
            attr_to_reduce = self._attributes
        for attr in attr_to_reduce:
            if isinstance(getattr(self, attr), pd.Series):
                setattr(
                    self,
                    attr,
                    getattr(self, attr).astype(to_type),
                )
            else:
                setattr(
                    self,
                    attr,
                    getattr(self, attr).apply(lambda _: _.astype(to_type)),
                )

    def to_csv(self, directory, reduce_memory=False, **kwargs):
        """
        Saves data in object to csv.

        Parameters
        ----------
        directory : str
            Directory to save data in.
        reduce_memory : bool, optional
            If True, size of time series data is reduced using
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory`.
            Optional parameters of
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory`
            can be passed as kwargs to this function. Default: False.

        Other Parameters
        ------------------
        kwargs :
            Kwargs may contain arguments of
            :attr:`~.network.overlying_grid.OverlyingGrid.reduce_memory`.

        """
        if reduce_memory is True:
            self.reduce_memory(**kwargs)

        os.makedirs(directory, exist_ok=True)

        for attr in self._attributes:
            if not getattr(self, attr).empty:
                if isinstance(getattr(self, attr), pd.Series):
                    getattr(self, attr).to_frame(name=attr).to_csv(
                        os.path.join(directory, f"{attr}.csv")
                    )
                else:
                    getattr(self, attr).to_csv(os.path.join(directory, f"{attr}.csv"))

    def from_csv(
        self,
        data_path,
        dtype=None,
        from_zip_archive=False,
        **kwargs,
    ):
        """
        Restores data in object from csv files.

        Parameters
        ----------
        data_path : str
            Path to directory to obtain data from. Must be a directory or zip
            archive.
        dtype : str, optional
            Numerical data type for data to be loaded from csv. E.g. "float32".
            Default: None.
        from_zip_archive : bool, optional
            Set True if data is archived in a zip archive. Default: False.

        """

        # get all attributes
        attrs = self._attributes

        if from_zip_archive:
            # read from zip archive
            # setup ZipFile Class
            zip = ZipFile(data_path)

            # get all directories and files within zip archive
            files = zip.namelist()

            # add directory and .csv to files to match zip archive
            attrs = {v: f"overlying_grid/{v}.csv" for v in attrs}

        else:
            # read from directory
            # check files within the directory
            files = os.listdir(data_path)

            # add .csv to files to match directory structure
            attrs = {v: f"{v}.csv" for v in attrs}

        attrs_to_read = {k: v for k, v in attrs.items() if v in files}

        for attr, file in attrs_to_read.items():
            if from_zip_archive:
                # open zip file to make it readable for pandas
                with zip.open(file) as f:
                    df = pd.read_csv(f, index_col=0, parse_dates=True, dtype=dtype)
            else:
                path = os.path.join(data_path, file)
                df = pd.read_csv(path, index_col=0, parse_dates=True, dtype=dtype)

            # convert to Series if Series is expected
            if isinstance(getattr(self, attr), pd.Series):
                df = df.squeeze("columns")

            setattr(self, attr, df)

        if from_zip_archive:
            # make sure to destroy ZipFile Class to close any open connections
            zip.close()

    def resample(self, method: str = "ffill", freq: str | pd.Timedelta = "15min"):
        """
        Resamples all time series to a desired resolution.

        See :attr:`~.EDisGo.resample_timeseries` for more information.

        Parameters
        ----------
        method : str, optional
            See :attr:`~.EDisGo.resample_timeseries` for more information.

        freq : str, optional
            See :attr:`~.EDisGo.resample_timeseries` for more information.

        """
        # get frequency of time series data
        timeindex = []
        for attr_str in self._attributes:
            attr = getattr(self, attr_str)
            if not attr.empty:
                if len(attr) >= 2:
                    timeindex = attr.index
                    break

        if len(timeindex) < 2:
            logger.warning(
                "Data cannot be resampled as it only contains one time step."
            )
            return

        freq_orig = timeindex[1] - timeindex[0]
        resample(self, freq_orig, method, freq)


def distribute_overlying_grid_requirements(edisgo_obj):
    """
    Distributes overlying grid requirements to components in grid.

    Overlying grid requirements for e.g. electromobility charging are distributed to
    all charging points where cars are parked, and for DSM to all DSM loads based
    on their available load increase and decrease at each time step.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        The eDisGo API object

    Returns
    --------
    :class:`~.EDisGo`
        New EDisGo object with only the topology data and adjusted time series data.

    """

    edisgo_copy = edisgo_obj.__class__()
    edisgo_copy.topology = deepcopy(edisgo_obj.topology)
    edisgo_copy.timeseries = deepcopy(edisgo_obj.timeseries)

    # electromobility - distribute charging time series from overlying grid to all
    # charging points based on upper power flexibility band
    if not edisgo_obj.overlying_grid.electromobility_active_power.empty:
        cp_loads = edisgo_obj.topology.loads_df.index[
            edisgo_obj.topology.loads_df.type == "charging_point"
        ]
        # scale flexibility band upper power timeseries
        scaling_df = edisgo_obj.electromobility.flexibility_bands[
            "upper_power"
        ].transpose() / edisgo_obj.electromobility.flexibility_bands["upper_power"].sum(
            axis=1
        )
        edisgo_copy.timeseries._loads_active_power.loc[:, cp_loads] = (
            scaling_df * edisgo_obj.overlying_grid.electromobility_active_power
        ).transpose()

    # storage units - distribute charging/discharging time series from overlying grid
    # to all storage units based on their installed capacity
    if not edisgo_obj.overlying_grid.storage_units_active_power.empty:
        scaling_factor = (
            edisgo_obj.topology.storage_units_df.p_nom
            / edisgo_obj.topology.storage_units_df.p_nom.sum()
        )
        scaling_df = pd.DataFrame(
            index=scaling_factor.index,
            columns=edisgo_copy.timeseries.timeindex,
            data=pd.concat(
                [scaling_factor] * len(edisgo_copy.timeseries.timeindex), axis=1
            ).values,
        )
        edisgo_copy.timeseries._storage_units_active_power = (
            scaling_df * edisgo_obj.overlying_grid.storage_units_active_power
        ).transpose()

    # central PtH - distribute dispatch time series from overlying grid
    # to all central PtH units based on their installed capacity
    if not edisgo_obj.overlying_grid.heat_pump_central_active_power.empty:
        hp_district = edisgo_obj.topology.loads_df[
            (edisgo_obj.topology.loads_df.type == "heat_pump")
            & (
                edisgo_obj.topology.loads_df.sector.isin(
                    ["district_heating", "district_heating_resistive_heater"]
                )
            )
        ]
        scaling_factor = hp_district.p_set / hp_district.p_set.sum()
        scaling_df = pd.DataFrame(
            index=scaling_factor.index,
            columns=edisgo_copy.timeseries.timeindex,
            data=pd.concat(
                [scaling_factor] * len(edisgo_copy.timeseries.timeindex), axis=1
            ).values,
        )
        edisgo_copy.timeseries._loads_active_power.loc[:, hp_district.index] = (
            scaling_df * edisgo_obj.overlying_grid.heat_pump_central_active_power
        ).transpose()

    # decentral PtH - distribute dispatch time series from overlying grid
    # to all decentral PtH units based on their installed capacity
    if not edisgo_obj.overlying_grid.heat_pump_decentral_active_power.empty:
        hp_individual = edisgo_obj.topology.loads_df.index[
            (edisgo_obj.topology.loads_df.type == "heat_pump")
            & (
                edisgo_obj.topology.loads_df.sector.isin(
                    ["individual_heating", "individual_heating_resistive_heater"]
                )
            )
        ]
        # scale with heat pump upper power
        scaling_factor = (
            edisgo_obj.topology.loads_df.p_set.loc[hp_individual]
            / edisgo_obj.topology.loads_df.p_set.loc[hp_individual].sum()
        )
        scaling_df = pd.DataFrame(
            index=scaling_factor.index,
            columns=edisgo_copy.timeseries.timeindex,
            data=pd.concat(
                [scaling_factor] * len(edisgo_copy.timeseries.timeindex), axis=1
            ).values,
        )
        edisgo_copy.timeseries._loads_active_power.loc[:, hp_individual] = (
            scaling_df * edisgo_obj.overlying_grid.heat_pump_decentral_active_power
        ).transpose()

    # DSM - distribute dispatch time series from overlying grid to all DSM loads based
    # on their maximum load increase (in case of positive dispatch values) or
    # their maximum load decrease (in case of negative dispatch values)
    if not edisgo_obj.overlying_grid.dsm_active_power.empty:
        dsm_loads = edisgo_obj.dsm.p_max.columns
        if len(dsm_loads) > 0:
            scaling_df_max = (
                edisgo_obj.dsm.p_max.transpose() / edisgo_obj.dsm.p_max.sum(axis=1)
            )
            scaling_df_min = (
                edisgo_obj.dsm.p_min.transpose() / edisgo_obj.dsm.p_min.sum(axis=1)
            )
            edisgo_copy.timeseries._loads_active_power.loc[:, dsm_loads] = (
                edisgo_obj.timeseries._loads_active_power.loc[:, dsm_loads]
                + (
                    scaling_df_min
                    * edisgo_obj.overlying_grid.dsm_active_power.clip(upper=0)
                ).transpose()
                + (
                    scaling_df_max
                    * edisgo_obj.overlying_grid.dsm_active_power.clip(lower=0)
                ).transpose()
            )
        else:
            logger.warning(
                "EDisGo object has no attribute 'dsm'. DSM timeseries from "
                "overlying grid cannot be distributed."
            )

    # curtailment
    if not edisgo_obj.overlying_grid.renewables_curtailment.empty:
        gens = edisgo_obj.topology.generators_df[
            edisgo_obj.topology.generators_df.type.isin(["solar", "wind"])
        ].index
        gen_per_ts = edisgo_obj.timeseries.generators_active_power.loc[:, gens].sum(
            axis=1
        )
        scaling_factor = (
            edisgo_obj.timeseries.generators_active_power.loc[:, gens].transpose()
            / gen_per_ts
        ).fillna(0)
        curtailment = (
            scaling_factor * edisgo_obj.overlying_grid.renewables_curtailment
        ).transpose()
        edisgo_copy.timeseries._generators_active_power.loc[:, gens] = (
            edisgo_obj.timeseries.generators_active_power.loc[:, gens] - curtailment
        )

    # reset reactive power time series
    edisgo_copy.set_time_series_reactive_power_control()
    return edisgo_copy
