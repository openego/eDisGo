import os
import logging
import pandas as pd
import pickle

from edisgo.network.topology import Topology
from edisgo.network.results import Results
from edisgo.network import timeseries
from edisgo.io import pypsa_io
from edisgo.tools import plots, tools
from edisgo.flex_opt.reinforce_grid import reinforce_grid
from edisgo.io.ding0_import import import_ding0_grid
from edisgo.io.generators_import import oedb as import_generators_oedb
from edisgo.tools.config import Config
from edisgo.tools.geo import find_nearest_bus
from edisgo.opf.run_mp_opf import run_mp_opf
from edisgo.opf.results.opf_result_class import OPFResults

if "READTHEDOCS" not in os.environ:
    from shapely.geometry import Point

logger = logging.getLogger("edisgo")


class EDisGo:
    """
    Provides the top-level API for invocation of data import, power flow
    analysis, network reinforcement, flexibility measures, etc..

    Parameters
    ----------
    ding0_grid : :obj:`str`
        Path to directory containing csv files of network to be loaded.
    generator_scenario : None or :obj:`str`, optional
        If None, the generator park of the imported grid is kept as is.
        Otherwise defines which scenario of future generator park to use
        and invokes grid integration of these generators. Possible options are
        'nep2035' and 'ego100'. These are scenarios from the research project
        `open_eGo <https://openegoproject.wordpress.com/>`_ (see
        `final report <https://www.uni-flensburg.de/fileadmin/content/\
        abteilungen/industrial/dokumente/downloads/veroeffentlichungen/\
        forschungsergebnisse/20190426endbericht-openego-fkz0325881-final.pdf>`_
        for more information on the scenarios).
        See :attr:`~.EDisGo.import_generators` for further information on how
        generators are integrated and what further options there are.
        Default: None.

    worst_case_analysis : None or :obj:`str`, optional
        If not None time series for feed-in and load will be generated
        according to the chosen worst case analysis.
        Possible options are:

        * 'worst-case'

          Feed-in and load for the two worst-case scenarios feed-in case and
          load case are generated.

        * 'worst-case-feedin'

          Feed-in and load for the worst-case scenario feed-in case is
          generated.

        * 'worst-case-load'

          Feed-in and load for the worst-case scenario load case is generated.

        Worst case scaling factors for loads and generators are specified in
        the config section `worst_case_scale_factor`.

        Be aware that if you choose to conduct a worst-case analysis your
        input for all other time series parameters (e.g.
        `timeseries_generation_fluctuating`,
        `timeseries_generation_dispatchable`,
        `timeseries_load`) will not be used.
        As eDisGo is designed to work with time series but worst cases
        are not time specific, a random time index 1/1/1970 is used.

        Default: None.

    timeseries_generation_fluctuating : :obj:`str` or \
    :pandas:`pandas.DataFrame<DataFrame>` or None, optional
        Parameter used to obtain time series for active power feed-in of
        fluctuating renewables wind and solar.
        Possible options are:

        * 'oedb'

          Hourly time series for the year 2011 are obtained from the OpenEnergy
          DataBase. See
          :func:`edisgo.io.timeseries_import.import_feedin_timeseries` for more
          information.

        * :pandas:`pandas.DataFrame<DataFrame>`

          DataFrame with time series for active power feed-in, normalized to
          a capacity of 1 MW.

          Time series can either be aggregated by technology type or by type
          and weather cell ID. In the first case columns of the DataFrame are
          'solar' and 'wind'; in the second case columns need to be a
          :pandas:`pandas.MultiIndex<MultiIndex>` with the first level
          containing the type and the second level the weather cell ID.

          Index needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

          When importing a ding0 grid and/or using predefined scenarios
          of the future generator park (see parameter `generator_scenario`),
          each generator has an assigned weather cell ID that identifies the
          weather data cell from the weather data set used in the research
          project `open_eGo <https://openegoproject.wordpress.com/>`_ to
          determine feed-in profiles. The weather cell ID can be retrieved
          from column `weather_cell_id` in
          :attr:`~.network.topology.Topology.generators_df` and could be
          overwritten to use own weather cells.

        Default: None.

    timeseries_generation_dispatchable : :pandas:`pandas.DataFrame<DataFrame>`\
    or None, optional
        DataFrame with time series for active power of each
        type of dispatchable generator, normalized to a capacity of 1 MW.

        Index needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

        Columns represent generator type (e.g. 'gas', 'coal', 'biomass').
        All in the current grid existing generator types can be retrieved
        from column `type` in
        :attr:`~.network.topology.Topology.generators_df`.
        Use 'other' if you don't want to explicitly provide every possible
        type.

        Default: None.
    timeseries_generation_reactive_power : \
    :pandas:`pandas.DataFrame<DataFrame>` or None, optional
        Dataframe with time series of normalized reactive power (normalized by
        the rated nominal active power) per technology and weather cell. Index
        needs to be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
        Columns represent generator type and can be a MultiIndex containing
        the weather cell ID in the second level. If the technology doesn't
        contain weather cell information, i.e. if it is not a solar
        or wind generator, this second level can be left as a numpy Nan or a
        None.

        If no time series for the technology or technology and weather cell ID
        is given, reactive power will be calculated from power factor and
        power factor mode in the config sections `reactive_power_factor` and
        `reactive_power_mode` and a warning will be raised.

        Default: None.
    timeseries_load : :obj:`str` or :pandas:`pandas.DataFrame<DataFrame>` or \
    None, optional
        Parameter used to obtain time series of active power of loads.
        Possible options are:

        * 'demandlib'

          Time series for the year specified in input parameter `timeindex` are
          generated using standard electric load profiles from the oemof
          `demandlib <https://github.com/oemof/demandlib/>`_.

        * :pandas:`pandas.DataFrame<DataFrame>`

          DataFrame with load time series of each type of load
          normalized with corresponding annual energy demand. Index needs to
          be a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.
          Columns represent load type. The in the current grid existing load
          types can be retrieved from column `sector` in
          :attr:`~.network.topology.Topology.loads_df`. In ding0 grids the
          differentiated sectors are 'residential', 'retail', 'industrial', and
          'agricultural'.

        Default: None.

    timeseries_load_reactive_power : :pandas:`pandas.DataFrame<DataFrame>` \
    or None, optional
        Dataframe with time series of normalized reactive power (normalized by
        annual energy demand) per load sector.

        Index needs to be a
        :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

        Columns represent load type. The in the current grid existing load
        types can be retrieved from column `sector` in
        :attr:`~.network.topology.Topology.loads_df`. In ding0 grids the
        differentiated sectors are 'residential', 'retail', 'industrial', and
        'agricultural'.

        If no time series for the load sector is given, reactive power will be
        calculated from power factor and power factor mode in the config
        sections `reactive_power_factor` and `reactive_power_mode` and a
        warning will be raised.

        Default: None.

    timeindex : None or :pandas:`pandas.DatetimeIndex<DatetimeIndex>`, optional
        Can be used to select time ranges of the feed-in and load time series
        that will be used in the power flow analysis. Also defines the year
        load time series are obtained for when choosing the 'demandlib' option
        to generate load time series.

    config_path : None or :obj:`str` or :obj:`dict`
        Path to the config directory. Options are:

        * None

          If `config_path` is None, configs are loaded from the edisgo
          default config directory ($HOME$/.edisgo). If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.

        * :obj:`str`

          If `config_path` is a string, configs will be loaded from the
          directory specified by `config_path`. If the directory
          does not exist it is created. If config files don't exist the
          default config files are copied into the directory.

        * :obj:`dict`

          A dictionary can be used to specify different paths to the
          different config files. The dictionary must have the following
          keys:

          * 'config_db_tables'
          * 'config_grid'
          * 'config_grid_expansion'
          * 'config_timeseries'

          Values of the dictionary are paths to the corresponding
          config file. In contrast to the other two options, the directories
          and config files must exist and are not automatically created.

        Default: None.

    Attributes
    ----------
    topology : :class:`~.network.topology.Topology`
        The topology is a container object holding the topology of the grids.
    timeseries: :class:`~.network.timeseries.TimeSeries`
        Container for component time series.
    results : :class:`~.network.results.Results`
        This is a container holding all calculation results from power flow
        analyses, curtailment, storage integration, etc.

    """

    def __init__(self, **kwargs):

        # load configuration
        self._config = Config(config_path=kwargs.get("config_path", None))

        # instantiate topology object and load grid data
        self.topology = Topology(config=self.config)
        self.import_ding0_grid(path=kwargs.get("ding0_grid", None))

        # set up results and time series container
        self.results = Results(self)
        self.opf_results = OPFResults()
        self.timeseries = timeseries.TimeSeries()

        # import new generators
        if kwargs.get("generator_scenario", None) is not None:
            self.import_generators(
                generator_scenario=kwargs.pop("generator_scenario"),
                **kwargs)

        # set up time series for feed-in and load
        # worst-case time series
        if kwargs.get("import_timeseries", True):
            if kwargs.get("worst_case_analysis", None):
                timeseries.get_component_timeseries(
                    edisgo_obj=self,
                    mode=kwargs.get("worst_case_analysis", None)
                )
            else:
                timeseries.get_component_timeseries(
                    edisgo_obj=self,
                    **kwargs
                )

    @property
    def config(self):
        """
        eDisGo configuration data.

        Returns
        -------
        :class:`~.tools.config.Config`
            Config object with configuration data from config files.

        """
        return self._config

    @config.setter
    def config(self, config_path):
        self._config = Config(config_path=config_path)

    def import_ding0_grid(self, path):
        """
        Import ding0 topology data from csv files in the format as
        `Ding0 <https://github.com/openego/ding0>`_ provides it.

        Parameters
        -----------
        path : str
            Path to directory containing csv files of network to be loaded.

        """
        if path is not None:
            import_ding0_grid(path, self)

    def to_pypsa(self, **kwargs):
        """
        Convert to PyPSA network representation.

        A network topology representation based on
        :pandas:`pandas.DataFrame<DataFrame>`. The overall container object of
        this data model, the :pypsa:`pypsa.Network<network>`,
        is set up.

        Parameters
        ----------
        kwargs :
            See :func:`~.io.pypsa_io.to_pypsa` for further information.

        Returns
        -------
        :pypsa:`pypsa.Network<network>`
            PyPSA network representation.

        """
        timesteps = kwargs.pop("timesteps", None)
        mode = kwargs.get("mode", None)

        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]
        # export grid
        if not mode:
            return pypsa_io.to_pypsa(self, timesteps, **kwargs)
        elif "mv" in mode:
            return pypsa_io.to_pypsa(
                self.topology.mv_grid, timesteps, **kwargs
            )
        elif mode == "lv":
            lv_grid_name = kwargs.get("lv_grid_name", None)
            if not lv_grid_name:
                raise ValueError(
                    "For exporting lv grids, name of lv_grid has "
                    "to be provided."
                )
            return pypsa_io.to_pypsa(
                self.topology._grids[lv_grid_name],
                mode=mode,
                timesteps=timesteps
            )
        else:
            raise ValueError("The entered mode is not a valid option.")

    def to_graph(self):
        """
        Returns graph representation of the grid.

        Returns
        -------
        :networkx:`networkx.Graph<network.Graph>`
            Graph representation of the grid as networkx Ordered Graph,
            where lines are represented by edges in the graph, and buses and
            transformers are represented by nodes.

        """

        return self.topology.to_graph()

    # def curtail(self, methodology, curtailment_timeseries, **kwargs):
    #     """
    #     Sets up curtailment time series.
    #
    #     Curtailment time series are written into
    #     :class:`~.network.network.TimeSeries`. See
    #     :class:`~.network.network.CurtailmentControl` for more information on
    #     parameters and methodologies.
    #
    #     # """
    #     raise NotImplementedError
    #     # CurtailmentControl(edisgo=self, methodology=methodology,
    #     #                    curtailment_timeseries=curtailment_timeseries,
    #     #                    mode=kwargs.pop('mode', None), **kwargs)

    def import_generators(self, generator_scenario=None,
                          **kwargs):
        """
        Gets generator park for specified scenario and integrates them into
        the grid.

        Currently, the only supported data source is scenario data generated
        in the research project
        `open_eGo <https://openegoproject.wordpress.com/>`_. You can choose
        between two scenarios: 'nep2035' and 'ego100'. You can get more
        information on the scenarios in the
        `final report <https://www.uni-flensburg.de/fileadmin/content/\
        abteilungen/industrial/dokumente/downloads/veroeffentlichungen/\
        forschungsergebnisse/20190426endbericht-openego-fkz0325881-final\
        .pdf>`_.

        The generator data is retrieved from the
        `open energy platform <https://openenergy-platform.org/>`_
        from tables for
        `conventional power plants <https://openenergy-platform.org/dataedit/\
        view/supply/ego_dp_conv_powerplant>`_ and
        `renewable power plants <https://openenergy-platform.org/dataedit/\
        view/supply/ego_dp_res_powerplant>`_.

        When the generator data is retrieved, the following steps are
        conducted:

            * Step 1: Update capacity of existing generators if `
              update_existing` is True, which it is by default.
            * Step 2: Remove decommissioned generators if
              `remove_decommissioned` is True, which it is by default.
            * Step 3: Integrate new MV generators.
            * Step 4: Integrate new LV generators.

        For more information on how generators are integrated, see
        :attr:`~.network.topology.Topology.connect_to_mv` and
        :attr:`~.network.topology.Topology.connect_to_lv`.

        After the generator park is changed there may be grid issues due to the
        additional in-feed. These are not solved automatically. If you want to
        have a stable grid without grid issues you can invoke the automatic
        grid expansion through the function :attr:`~.EDisGo.reinforce`.

        Parameters
        ----------
        generator_scenario : str
            Scenario for which to retrieve generator data. Possible options
            are 'nep2035' and 'ego100'.

        Other Parameters
        ----------------
        kwargs :
            See :func:`edisgo.io.generators_import.oedb`.

        """
        import_generators_oedb(edisgo_object=self,
                               generator_scenario=generator_scenario,
                               **kwargs)

    def analyze(self, mode=None, timesteps=None, **kwargs):
        """Conducts a static, non-linear power flow analysis

        Conducts a static, non-linear power flow analysis using
        `PyPSA <https://www.pypsa.org/doc/power_flow.html#full-non-linear-power-flow>`_
        and writes results (active, reactive and apparent power as well as
        current on lines and voltages at buses) to
        :class:`~.network.results.Results`
        (e.g. :attr:`~.network.results.Results.v_res` for voltages).
        See :func:`~.io.pypsa_io.to_pypsa` for more information.

        Parameters
        ----------
        mode : str
            Allows to toggle between power flow analysis (PFA) on the whole
            network topology (default: None), only MV ('mv' or 'mvlv') or only
            LV ('lv'). Defaults to None which equals power flow analysis for
            MV + LV.
        timesteps : :pandas:`pandas.DatetimeIndex<DatetimeIndex>` or \
            :pandas:`pandas.Timestamp<Timestamp>`
            Timesteps specifies for which time steps to conduct the power flow
            analysis. It defaults to None in which case the time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex` are
            used.

        """
        if timesteps is None:
            timesteps = self.timeseries.timeindex
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timesteps, "__len__"):
            timesteps = [timesteps]

        pypsa_network = self.to_pypsa(mode=mode, timesteps=timesteps, **kwargs)

        # run power flow analysis
        pf_results = pypsa_network.pf(
            timesteps, use_seed=kwargs.get("use_seed", False))

        if all(pf_results["converged"]["0"].tolist()):
            pypsa_io.process_pfa_results(self, pypsa_network, timesteps)
        else:
            raise ValueError("Power flow analysis did not converge for the "
                             "following time steps: {}.".format(
                timesteps[~pf_results["converged"]["0"]].tolist())
            )

    def reinforce(self, **kwargs):
        """
        Reinforces the network and calculates network expansion costs.

        See :func:`edisgo.flex_opt.reinforce_grid.reinforce_grid` for more
        information.

        """
        results = reinforce_grid(
            self,
            max_while_iterations=kwargs.get("max_while_iterations", 10),
            copy_grid=kwargs.get("copy_grid", False),
            timesteps_pfa=kwargs.get("timesteps_pfa", None),
            combined_analysis=kwargs.get("combined_analysis", False),
            mode=kwargs.get("mode", None),
        )

        # add measure to Results object
        if not kwargs.get("copy_grid", False):
            self.results.measures = "grid_expansion"

        return results

    def perform_mp_opf(self, timesteps, storage_series=[], **kwargs):
        """
        Run optimal power flow with julia.

        Parameters
        -----------
        timesteps : list
            List of timesteps to perform OPF for.
        kwargs :
            See :func:`~.opf.run_mp_opf.run_mp_opf` for further
            information.

        Returns
        --------
        str
            Status of optimization.

        """
        status = run_mp_opf(
            self, timesteps, storage_series=storage_series, **kwargs
        )
        return status

    def aggregate_components(self, mode="by_component_type",
                             aggregate_generators_by_cols=["bus"],
                             aggregate_loads_by_cols=["bus"],
                             aggregate_charging_points_by_cols=["bus"]):
        """
        Aggregates generators, loads and charging points at the same bus.

        There are several options how to aggregate. By default all components
        of the same type are aggregated separately. You can specify further
        columns to consider in the aggregation, such as the generator type
        or the load sector.

        Be aware that by aggregating components you lose some information
        e.g. on load sector or charging point use case.

        Parameters
        -----------
        mode : str
            Valid options are 'by_component_type' and 'by_load_and_generation'.
            In case of aggregation 'by_component_type' generators, loads and
            charging points are aggregated separately, by the respectively
            specified columns, given in `aggregate_generators_by_cols`,
            `aggregate_loads_by_cols`, and `aggregate_charging_points_by_cols`.
            In case of aggregation 'by_load_and_generation', all loads and
            charging points at the same bus are aggregated. Input in
            `aggregate_loads_by_cols` and `aggregate_charging_points_by_cols`
            is ignored. Generators are aggregated by the columns specified in
            `aggregate_generators_by_cols`.
        aggregate_generators_by_cols : list(str)
            List of columns to aggregate generators at the same bus by. Valid
            columns are all columns in
            :attr:`~.network.topology.Topology.generators_df`.
        aggregate_loads_by_cols : list(str)
            List of columns to aggregate loads at the same bus by. Valid
            columns are all columns in
            :attr:`~.network.topology.Topology.loads_df`.
        aggregate_charging_points_by_cols : list(str)
            List of columns to aggregate charging points at the same bus by.
            Valid columns are all columns in
            :attr:`~.network.topology.Topology.charging_points_df`.

        """
        # aggregate generators at the same bus
        if mode is "by_component_type" or "by_load_and_generation":
            if not self.topology.generators_df.empty:
                gens_groupby = self.topology.generators_df.groupby(
                    aggregate_generators_by_cols)
                naming = "Generators_{}"
                # set up new generators_df
                gens_df_grouped = gens_groupby.sum().reset_index()
                gens_df_grouped["name"] = gens_df_grouped.apply(
                    lambda _: naming.format(
                        "_".join(_.loc[aggregate_generators_by_cols])),
                    axis=1)
                gens_df_grouped["control"] = "PQ"
                gens_df_grouped["control"] = "misc"
                if "weather_cell_id" in gens_df_grouped.columns:
                    gens_df_grouped.drop(
                        columns=["weather_cell_id"], inplace=True)
                self.topology.generators_df = gens_df_grouped.set_index("name")
                # set up new generator time series
                groups = gens_groupby.groups
                if isinstance(list(groups.keys())[0], tuple):
                    self.timeseries.generators_active_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format("_".join(k)):
                                 self.timeseries.generators_active_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                    self.timeseries.generators_reactive_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format("_".join(k)):
                                 self.timeseries.generators_reactive_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                else:
                    self.timeseries.generators_active_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format(k):
                                 self.timeseries.generators_active_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                    self.timeseries.generators_reactive_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format(k):
                                 self.timeseries.generators_reactive_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)

        # aggregate conventional loads at the same bus and charging points
        # at the same bus separately
        if mode is "by_component_type":

            # conventional loads
            if not self.topology.loads_df.empty:
                loads_groupby = self.topology.loads_df.groupby(
                    aggregate_loads_by_cols)
                naming = "Loads_{}"
                # set up new loads_df
                loads_df_grouped = loads_groupby.sum().reset_index()
                loads_df_grouped["name"] = loads_df_grouped.apply(
                    lambda _: naming.format(
                        "_".join(_.loc[aggregate_loads_by_cols])),
                    axis=1)
                self.topology.loads_df = loads_df_grouped.set_index("name")
                # set up new loads time series
                groups = loads_groupby.groups
                if isinstance(list(groups.keys())[0], tuple):
                    self.timeseries.loads_active_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format("_".join(k)):
                                 self.timeseries.loads_active_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                    self.timeseries.loads_reactive_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format("_".join(k)):
                                 self.timeseries.loads_reactive_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                else:
                    self.timeseries.loads_active_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format(k):
                                 self.timeseries.loads_active_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                    self.timeseries.loads_reactive_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format(k):
                                 self.timeseries.loads_reactive_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)

            # charging points
            if not self.topology.charging_points_df.empty:
                loads_groupby = self.topology.charging_points_df.groupby(
                    aggregate_charging_points_by_cols)
                naming = "Charging_points_{}"
                # set up new charging_points_df
                loads_df_grouped = loads_groupby.sum().reset_index()
                loads_df_grouped["name"] = loads_df_grouped.apply(
                    lambda _: naming.format(
                        "_".join(_.loc[aggregate_charging_points_by_cols])),
                    axis=1)
                self.topology.charging_points_df = loads_df_grouped.set_index(
                    "name")
                # set up new charging points time series
                groups = loads_groupby.groups
                if isinstance(list(groups.keys())[0], tuple):
                    self.timeseries.charging_points_active_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format("_".join(k)):
                                 self.timeseries.charging_points_active_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                    self.timeseries.charging_points_reactive_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format("_".join(k)):
                                 self.timeseries.charging_points_reactive_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                else:
                    self.timeseries.charging_points_active_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format(k):
                                 self.timeseries.charging_points_active_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)
                    self.timeseries.charging_points_reactive_power = pd.concat(
                        [pd.DataFrame(
                            {naming.format(k):
                                 self.timeseries.charging_points_reactive_power.loc[
                                 :, v].sum(axis=1)})
                            for k, v in groups.items()], axis=1)

        # aggregate all loads (conventional loads and charging points) at the
        # same bus
        elif mode is "by_load_and_generation":
            aggregate_loads_by_cols = ["bus"]
            loads_groupby = pd.concat(
                [self.topology.loads_df.loc[:, ["bus", "peak_load"]],
                 self.topology.charging_points_df.loc[
                 :, ["bus", "p_nom"]].rename(columns={"p_nom": "peak_load"})]
            ).groupby(aggregate_loads_by_cols)
            naming = "Loads_{}"
            # set up new loads_df
            loads_df_grouped = loads_groupby.sum().reset_index()
            loads_df_grouped["name"] = loads_df_grouped.apply(
                lambda _: naming.format(
                    "_".join(_.loc[aggregate_loads_by_cols])),
                axis=1)
            self.topology.loads_df = loads_df_grouped.set_index("name")
            # set up new loads time series
            groups = loads_groupby.groups
            ts_active = pd.concat([
                self.timeseries.loads_active_power,
                self.timeseries.charging_points_active_power],
                axis=1)
            ts_reactive = pd.concat([
                self.timeseries.loads_reactive_power,
                self.timeseries.charging_points_reactive_power],
                axis=1)
            if isinstance(list(groups.keys())[0], tuple):

                self.timeseries.loads_active_power = pd.concat(
                    [pd.DataFrame(
                        {naming.format("_".join(k)):
                             ts_active.loc[:, v].sum(axis=1)})
                        for k, v in groups.items()], axis=1)
                self.timeseries.loads_reactive_power = pd.concat(
                    [pd.DataFrame(
                        {naming.format("_".join(k)):
                             ts_reactive.loc[:, v].sum(axis=1)})
                        for k, v in groups.items()], axis=1)
            else:
                self.timeseries.loads_active_power = pd.concat(
                    [pd.DataFrame(
                        {naming.format(k):
                             ts_active.loc[:, v].sum(axis=1)})
                        for k, v in groups.items()], axis=1)
                self.timeseries.loads_reactive_power = pd.concat(
                    [pd.DataFrame(
                        {naming.format(k):
                             ts_reactive.loc[:, v].sum(axis=1)})
                        for k, v in groups.items()], axis=1)
            # overwrite charging points
            self.topology.charging_points_df = pd.DataFrame(
                columns=["bus", "p_nom", "use_case"])
            self.timeseries.charging_points_active_power = pd.DataFrame(
                index=self.timeseries.timeindex)
            self.timeseries.charging_points_reactive_power = pd.DataFrame(
                index=self.timeseries.timeindex)

    def plot_mv_grid_topology(self, technologies=False, **kwargs):
        """
        Plots plain MV network topology and optionally nodes by technology type
        (e.g. station or generator).

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        Parameters
        ----------
        technologies : :obj:`Boolean`
            If True plots stations, generators, etc. in the topology in
            different colors. If False does not plot any nodes. Default: False.

        """

        plots.mv_grid_topology(
            self,
            node_color="technology" if technologies is True else None,
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            title=kwargs.get("title", ""),
        )

    def plot_mv_voltages(self, **kwargs):
        """
        Plots voltages in MV network on network topology plot.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.v_res is None:
                logging.warning(
                    "Voltages from power flow "
                    "analysis must be available to plot them."
                )
                return
        except AttributeError:
            logging.warning(
                "Results must be available to plot voltages. "
                "Please analyze grid first."
            )
            return
        except ValueError:
            pass

        plots.mv_grid_topology(
            self,
            timestep=kwargs.get("timestep", None),
            node_color="voltage",
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            voltage=self.results.v_res,
            limits_cb_nodes=kwargs.get("limits_cb_nodes", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            title=kwargs.get("title", ""),
        )

    def plot_mv_line_loading(self, **kwargs):
        """
        Plots relative line loading (current from power flow analysis to
        allowed current) of MV lines.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.i_res is None:
                logging.warning(
                    "Currents `i_res` from power flow analysis "
                    "must be available to plot line loading."
                )
                return
        except AttributeError:
            logging.warning(
                "Results must be available to plot line loading. "
                "Please analyze grid first."
            )
            return

        plots.mv_grid_topology(
            self,
            timestep=kwargs.get("timestep", None),
            line_color="loading",
            node_color=kwargs.get("node_color", None),
            line_load=self.results.i_res,
            filename=kwargs.get("filename", None),
            arrows=kwargs.get("arrows", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            voltage=self.results.v_res,
            limits_cb_lines=kwargs.get("limits_cb_lines", None),
            limits_cb_nodes=kwargs.get("limits_cb_nodes", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            lines_cmap=kwargs.get("lines_cmap", "inferno_r"),
            title=kwargs.get("title", ""),
            scaling_factor_line_width=kwargs.get(
                "scaling_factor_line_width", None
            ),
            curtailment_df=kwargs.get("curtailment_df", None),
        )

    def plot_mv_grid_expansion_costs(self, **kwargs):
        """
        Plots grid expansion costs per MV line.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        try:
            if self.results.grid_expansion_costs is None:
                logging.warning(
                    "Grid expansion cost results needed to plot "
                    "them. Please do grid reinforcement."
                )
                return
        except AttributeError:
            logging.warning(
                "Results of MV topology needed to  plot topology "
                "expansion costs. Please reinforce first."
            )
            return

        plots.mv_grid_topology(
            self,
            line_color="expansion_costs",
            grid_expansion_costs=self.results.grid_expansion_costs,
            filename=kwargs.get("filename", None),
            grid_district_geom=kwargs.get("grid_district_geom", True),
            background_map=kwargs.get("background_map", True),
            limits_cb_lines=kwargs.get("limits_cb_lines", None),
            xlim=kwargs.get("xlim", None),
            ylim=kwargs.get("ylim", None),
            lines_cmap=kwargs.get("lines_cmap", "inferno_r"),
            title=kwargs.get("title", ""),
            scaling_factor_line_width=kwargs.get(
                "scaling_factor_line_width", None
            ),
        )

    def plot_mv_storage_integration(self, **kwargs):
        """
        Plots storage position in MV topology of integrated storage units.

        For more information see :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        plots.mv_grid_topology(
            self, node_color="storage_integration", **kwargs
        )

    def plot_mv_grid(self, **kwargs):
        """
        General plotting function giving all options of function
        :func:`edisgo.tools.plots.mv_grid_topology`.

        """
        plots.mv_grid_topology(self, **kwargs)

    def histogram_voltage(self, timestep=None, title=True, **kwargs):
        """
        Plots histogram of voltages.

        For more information on the histogram plot and possible configurations
        see :func:`edisgo.tools.plots.histogram`.

        Parameters
        ----------
        timestep : :pandas:`pandas.Timestamp<Timestamp>` or list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
            Specifies time steps histogram is plotted for. If timestep is None
            all time steps voltages are calculated for are used. Default: None.
        title : :obj:`str` or :obj:`bool`, optional
            Title for plot. If True title is auto generated. If False plot has
            no title. If :obj:`str`, the provided title is used. Default: True.

        """
        try:
            data = self.results.v_res
            if data is None:
                logger.warning(
                    "Results for voltages are required for "
                    "voltage histogramm. Please analyze first."
                )
                return
        except AttributeError:
            logger.warning(
                "Results are required for "
                "voltage histogramm. Please analyze first."
            )
            return

        if timestep is None:
            timestep = data.index
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timestep, "__len__"):
            timestep = [timestep]

        if title is True:
            if len(timestep) == 1:
                title = "Voltage histogram for time step {}".format(
                    timestep[0]
                )
            else:
                title = "Voltage histogram \nfor time steps {} to {}".format(
                    timestep[0], timestep[-1]
                )
        elif title is False:
            title = None
        plots.histogram(data=data, title=title, timeindex=timestep, **kwargs)

    def histogram_relative_line_load(
        self, timestep=None, title=True, voltage_level="mv_lv", **kwargs
    ):
        """
        Plots histogram of relative line loads.

        For more information on how the relative line load is calculated see
        :func:`edisgo.tools.tools.get_line_loading_from_network`.
        For more information on the histogram plot and possible configurations
        see :func:`edisgo.tools.plots.histogram`.

        Parameters
        ----------
        timestep : :pandas:`pandas.Timestamp<Timestamp>` or list(:pandas:`pandas.Timestamp<Timestamp>`) or None, optional
            Specifies time step(s) histogram is plotted for. If `timestep` is
            None all time steps currents are calculated for are used.
            Default: None.
        title : :obj:`str` or :obj:`bool`, optional
            Title for plot. If True title is auto generated. If False plot has
            no title. If :obj:`str`, the provided title is used. Default: True.
        voltage_level : :obj:`str`
            Specifies which voltage level to plot voltage histogram for.
            Possible options are 'mv', 'lv' and 'mv_lv'. 'mv_lv' is also the
            fallback option in case of wrong input. Default: 'mv_lv'

        """
        try:
            if self.results.i_res is None:
                logger.warning(
                    "Currents `i_res` from power flow analysis "
                    "must be available to plot histogram line "
                    "loading."
                )
                return
        except AttributeError:
            logger.warning(
                "Results must be available to plot histogram line "
                "loading. Please analyze grid first."
            )
            return

        if voltage_level == "mv":
            lines = self.topology.mv_grid.lines_df
        elif voltage_level == "lv":
            lines = self.topology.lines_df[
                ~self.topology.lines_df.index.isin(
                    self.topology.mv_grid.lines_df.index)
            ]
        else:
            lines = self.topology.lines_df

        rel_line_loading = tools.calculate_relative_line_load(
            self, lines.index, timestep
        )

        if timestep is None:
            timestep = rel_line_loading.index
        # check if timesteps is array-like, otherwise convert to list
        if not hasattr(timestep, "__len__"):
            timestep = [timestep]

        if title is True:
            if len(timestep) == 1:
                title = "Relative line load histogram for time step {}".format(
                    timestep[0]
                )
            else:
                title = (
                    "Relative line load histogram \nfor time steps "
                    "{} to {}".format(timestep[0], timestep[-1])
                )
        elif title is False:
            title = None
        plots.histogram(data=rel_line_loading, title=title, **kwargs)

    def save(
        self,
        directory,
        save_results=True,
        save_topology=True,
        save_timeseries=True,
        **kwargs
    ):
        """
        Saves EDisGo object to csv.

        It can be chosen if results, topology and timeseries should be saved.
        For each one, a separate directory is created.

        Parameters
        ----------
        directory : str
            Main directory to save EDisGo object to.
        save_results : bool, optional
            Indicates whether to save :class:`~.network.results.Results`
            object. Per default it is saved. See
            :attr:`~.network.results.Results.to_csv` for more information.
        save_topology : bool, optional
            Indicates whether to save :class:`~.network.topology.Topology`.
            Per default it is saved. See
            :attr:`~.network.topology.Topology.to_csv` for more information.
        save_timeseries : bool, optional
            Indicates whether to save :class:`~.network.timeseries.Timeseries`.
            Per default it is saved. See
            :attr:`~.network.timeseries.Timeseries.to_csv` for more
            information.

        Other Parameters
        ------------------
        reduce_memory : bool, optional
            If True, size of dataframes containing time series in
            :class:`~.network.results.Results` and
            :class:`~.network.timeseries.TimeSeries`
            is reduced. See :attr:`~.network.results.Results.reduce_memory`
            and :attr:`~.network.timeseries.TimeSeries.reduce_memory` for more
            information. Type to convert to can be specified by providing
            `to_type` as keyword argument. Further parameters of reduce_memory
            functions cannot be passed here. Call these functions directly to
            make use of further options. Default: False.
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".

        """
        os.makedirs(directory, exist_ok=True)
        if save_results:
            self.results.to_csv(
                os.path.join(directory, "results"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32"),
                parameters=kwargs.get("parameters", None)
            )
        if save_topology:
            self.topology.to_csv(
                os.path.join(directory, "topology")
            )
        if save_timeseries:
            self.timeseries.to_csv(
                os.path.join(directory, "timeseries"),
                reduce_memory=kwargs.get("reduce_memory", False),
                to_type=kwargs.get("to_type", "float32")
            )

    def add_component(
        self,
        comp_type,
        add_ts=True,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs
    ):
        """
        Adds single component to network topology.

        Components can be lines or buses as well as generators, loads,
        charging points or storage units.

        Parameters
        ----------
        comp_type : str
            Type of added component. Can be 'Bus', 'Line', 'Load', 'Generator',
            'StorageUnit', 'Transformer' or 'ChargingPoint'.
        add_ts : bool
            Indicator if time series for component are added as well.
        ts_active_power : :pandas:`pandas.Series<series>`
            Active power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are active power per time step in MW.
        ts_reactive_power : :pandas:`pandas.Series<series>`
            Reactive power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are reactive power per time step in MVA.
        **kwargs: dict
            Attributes of added component. See respective functions for required
            entries. For 'Load', 'Generator' and 'StorageUnit' the boolean
            add_ts determines whether a time series is created for the new
            component or not.

        Todo: change into add_components to allow adding of several components
            at a time, change topology.add_load etc. to add_loads, where
            lists of parameters can be inserted
        """
        if comp_type == "Bus":
            comp_name = self.topology.add_bus(**kwargs)
        elif comp_type == "Line":
            comp_name = self.topology.add_line(**kwargs)
        elif comp_type == "Load" or comp_type == "charging_park":
            comp_name = self.topology.add_load(**kwargs)
            if add_ts:
                timeseries.add_loads_timeseries(
                    edisgo_obj=self, load_names=comp_name, **kwargs
                )

        elif comp_type == "Generator":
            comp_name = self.topology.add_generator(**kwargs)
            if add_ts:
                timeseries.add_generators_timeseries(
                    edisgo_obj=self, generator_names=comp_name, **kwargs
                )

        elif comp_type == "ChargingPoint":
            comp_name = self.topology.add_charging_point(**kwargs)
            if add_ts:
                if ts_active_power is not None and ts_reactive_power is not None:
                    timeseries.add_charging_points_timeseries(
                        self, [comp_name],
                        ts_active_power=pd.DataFrame({
                            comp_name: ts_active_power}),
                        ts_reactive_power=pd.DataFrame({
                            comp_name: ts_reactive_power})
                    )
                else:
                    raise ValueError("Time series for charging points need "
                                     "to be provided.")

        elif comp_type == "StorageUnit":
            comp_name = self.topology.add_storage_unit(
                **kwargs,
            )
            if add_ts:
                if isinstance(ts_active_power, pd.Series):
                    ts_active_power = pd.DataFrame(
                        {comp_name: ts_active_power}
                    )
                if isinstance(ts_reactive_power, pd.Series):
                    ts_reactive_power = pd.DataFrame(
                        {comp_name: ts_reactive_power}
                    )
                timeseries.add_storage_units_timeseries(
                    edisgo_obj=self,
                    storage_unit_names=comp_name,
                    timeseries_storage_units=ts_active_power,
                    timeseries_storage_units_reactive_power=ts_reactive_power,
                    **kwargs
                )
        else:
            raise ValueError("Component type is not correct.")
        return comp_name

    def integrate_component(
        self,
        comp_type,
        geolocation,
        voltage_level=None,
        add_ts=True,
        ts_active_power=None,
        ts_reactive_power=None,
        **kwargs
    ):
        """
        Adds single component to topology based on geolocation.

        Currently components can be generators or charging points.

        Parameters
        ----------
        comp_type : str
            Type of added component. Can be 'Generator' or 'ChargingPoint'.
        geolocation : :shapely:`shapely.Point<Point>` or tuple
            Geolocation of the new component. In case of tuple, the geolocation
            must be given in the form (longitude, latitude).
        voltage_level : int, optional
            Specifies the voltage level the new component is integrated in.
            Possible options are 4 (MV busbar), 5 (MV grid), 6 (LV busbar) or
            7 (LV grid). If no voltage level is provided the voltage level
            is determined based on the nominal power `p_nom` (given as kwarg)
            as follows:

            * voltage level 4 (MV busbar): nominal power between 4.5 MW and
              17.5 MW
            * voltage level 5 (MV grid) : nominal power between 0.3 MW and
              4.5 MW
            * voltage level 6 (LV busbar): nominal power between 0.1 MW and
              0.3 MW
            * voltage level 7 (LV grid): nominal power below 0.1 MW

        add_ts : bool, optional
            Indicator if time series for component are added as well.
            Default: True.
        ts_active_power : :pandas:`pandas.Series<Series>`, optional
            Active power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are active power per time step in MW. Currently, if you want
            to add time series (if `add_ts` is True), you must provide a
            time series. It is not automatically retrieved.
        ts_reactive_power : :pandas:`pandas.Series<Series>`, optional
            Reactive power time series of added component. Index of the series
            must contain all time steps in
            :attr:`~.network.timeseries.TimeSeries.timeindex`.
            Values are reactive power per time step in MVA. Currently, if you
            want to add time series (if `add_ts` is True), you must provide a
            time series. It is not automatically retrieved.

        Other Parameters
        ------------------
        kwargs :
            Attributes of added component.
            See :attr:`~.network.topology.Topology.add_generator` respectively
            :attr:`~.network.topology.Topology.add_charging_point` methods
            for more information on required and optional parameters of
            generators and charging points.

        """
        supported_voltage_levels = {4, 5, 6, 7}
        p_nom = kwargs.get('p_nom', None)
        if voltage_level not in supported_voltage_levels:
            if p_nom is None:
                raise ValueError(
                    "Neither appropriate voltage level nor nominal power "
                    "were supplied.")
            # Determine voltage level manually from nominal power
            if 4.5 < p_nom <= 17.5:
                voltage_level = 4
            elif 0.3 < p_nom <= 4.5:
                voltage_level = 5
            elif 0.1 < p_nom <= 0.3:
                voltage_level = 6
            elif 0 < p_nom <= 0.1:
                voltage_level = 7
            else:
                raise ValueError("Unsupported voltage level")

        # check if geolocation is given as shapely Point, otherwise transform
        # to shapely Point
        if not type(geolocation) is Point:
            geolocation = Point(geolocation)

        # Connect in MV
        if voltage_level in [4, 5]:
            kwargs['voltage_level'] = voltage_level
            kwargs['geom'] = geolocation
            comp_name = self.topology.connect_to_mv(
                self, kwargs, comp_type)

        # Connect in LV
        else:
            substations = self.topology.buses_df.loc[
                self.topology.transformers_df.bus1.unique()]
            nearest_substation, _ = find_nearest_bus(geolocation, substations)
            kwargs['mvlv_subst_id'] = int(nearest_substation.split("_")[-2])
            kwargs['geom'] = geolocation
            kwargs['voltage_level'] = voltage_level
            comp_name = self.topology.connect_to_lv(self, kwargs, comp_type)

        if add_ts:
            if comp_type == 'Generator':
                # ToDo: Adding time series for generators manually does
                #   currently not work
                func = timeseries.add_generators_timeseries
            else:
                func = timeseries.add_charging_points_timeseries
            func(
                self, [comp_name],
                ts_active_power=pd.DataFrame({
                    comp_name: ts_active_power}),
                ts_reactive_power=pd.DataFrame({
                    comp_name: ts_reactive_power})
            )

        return comp_name

    def remove_component(self, comp_type, comp_name, drop_ts=True):
        """
        Removes single component from respective DataFrame. If drop_ts is set
        to True, timeseries of elements are deleted as well.

        Parameters
        ----------
        comp_type: str
            Type of removed component. Can be 'Bus', 'Line', 'Load',
            'Generator', 'StorageUnit', 'Transformer'.
        comp_name: str
            Name of component to be removed.
        drop_ts: Boolean
            Indicator if timeseries for component are removed as well. Defaults
            to True.

        Todo: change into remove_components, when add_component is changed into
            add_components, to allow removal of several components at a time

        """
        if comp_type == "Bus":
            self.topology.remove_bus(comp_name)
        elif comp_type == "Line":
            self.topology.remove_line(comp_name)
        elif comp_type == "Load":
            self.topology.remove_load(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self, comp_type="loads", comp_names=comp_name
                )

        elif comp_type == "Generator":
            self.topology.remove_generator(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self,
                    comp_type="generators",
                    comp_names=comp_name,
                )
        elif comp_type == "StorageUnit":
            self.topology.remove_storage_unit(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self,
                    comp_type="storage_units",
                    comp_names=comp_name,
                )
        elif comp_type == "ChargingPoint":
            self.topology.remove_charging_point(comp_name)
            if drop_ts:
                timeseries._drop_existing_component_timeseries(
                    edisgo_obj=self,
                    comp_type="charging_points",
                    comp_names=comp_name
                )
        else:
            raise ValueError("Component type is not correct.")

    def save_edisgo_to_pickle(self, path='', filename=None):
        abs_path = os.path.abspath(path)
        if filename is None:
            filename = "edisgo_object_{ext}.pkl".format(
                ext=self.topology.mv_grid.id)
        pickle.dump(self, open(os.path.join(abs_path, filename), "wb"))

    def reduce_memory(self, **kwargs):
        """
        Reduces size of dataframes containing time series to save memory.

        Per default, float data is stored as float64. As this precision is
        barely needed, this function can be used to convert time series data
        to a data subtype with less memory usage, such as float32.

        Other Parameters
        -----------------
        to_type : str, optional
            Data type to convert time series data to. This is a tradeoff
            between precision and memory. Default: "float32".
        results_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.results.Results.reduce_memory` for more
            information.
        timeseries_attr_to_reduce : list(str), optional
            See `attr_to_reduce` parameter in
            :attr:`~.network.timeseries.TimeSeries.reduce_memory` for more
            information.

        """
        # time series
        self.timeseries.reduce_memory(
            to_type=kwargs.get("to_type", "float32"),
            attr_to_reduce=kwargs.get("timeseries_attr_to_reduce", None)
        )
        # results
        self.results.reduce_memory(
            to_type=kwargs.get("to_type", "float32"),
            attr_to_reduce=kwargs.get("results_attr_to_reduce", None)
        )


def import_edisgo_from_pickle(filename, path=''):
    abs_path = os.path.abspath(path)
    return pickle.load(open(os.path.join(abs_path, filename), "rb"))


def import_edisgo_from_files(directory="", import_topology=True,
                             import_timeseries=False, import_results=False,
                             **kwargs):
    edisgo_obj = EDisGo(import_timeseries=False)
    if import_topology:
        topology_dir = kwargs.get("topology_directory",
                                  os.path.join(directory, "topology"))
        if os.path.exists(topology_dir):
            edisgo_obj.topology.from_csv(topology_dir, edisgo_obj)
        else:
            logging.warning(
                'No topology directory found. Topology not imported.')
    if import_timeseries:
        if os.path.exists(os.path.join(directory, "timeseries")):
            edisgo_obj.timeseries.from_csv(os.path.join(directory, "timeseries"))
        else:
            logging.warning(
                'No timeseries directory found. Timeseries not imported.')
    if import_results:
        parameters = kwargs.get('parameters', None)
        if os.path.exists(os.path.join(directory, "results")):
            edisgo_obj.results.from_csv(os.path.join(directory, "results"),
                                        parameters)
        else:
            logging.warning('No results directory found. Results not imported.')
    if kwargs.get('import_residual_load', False):
        if os.path.exists(
                os.path.join(directory, 'time_series_sums.csv')):
            residual_load = pd.read_csv(
                os.path.join(directory, 'time_series_sums.csv')).rename(
                columns={'Unnamed: 0': 'timeindex'}).set_index('timeindex')['residual_load']
            residual_load.index = pd.to_datetime(residual_load.index)
            edisgo_obj.timeseries._residual_load = residual_load
    return edisgo_obj
