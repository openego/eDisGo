import os
import logging
import csv
import pandas as pd
import numpy as np
from math import sqrt

from edisgo.network.grids import MVGrid

logger = logging.getLogger("edisgo")


class Results:
    """
    Power flow analysis results management

    Includes raw power flow analysis results, history of measures to increase
    the network's hosting capacity and information about changes of equipment.

    Attributes
    ----------
    edisgo_object : :class:`~.EDisGo`

    """

    def __init__(self, edisgo_object):
        self.edisgo_object = edisgo_object
        self._measures = ["original"]
        self._pfa_p = None
        self._pfa_q = None
        self._i_res = None
        self._v_res = None
        self._equipment_changes = pd.DataFrame()
        self._grid_expansion_costs = None
        self._grid_losses = None
        self._hv_mv_exchanges = None
        self._curtailment = None
        self._storage_integration = None
        self._unresolved_issues = {}
        self._storage_units_costs_reduction = None

    @property
    def measures(self):
        """
        List with the history of measures to increase network's hosting capacity.

        Parameters
        ----------
        measure : :obj:`str`
            Measure to increase network's hosting capacity. Possible options are
            'grid_expansion', 'storage_integration', 'curtailment'.

        Returns
        -------
        measures : :obj:`list`
            A stack that details the history of measures to increase network's
            hosting capacity. The last item refers to the latest measure. The
            key `original` refers to the state of the network topology as it was
            initially imported.

        """
        return self._measures

    @measures.setter
    def measures(self, measure):
        self._measures.append(measure)

    @property
    def pfa_p(self):
        """
        Active power results from power flow analysis in kW.

        Holds power flow analysis results for active power for the last
        iteration step. Index of the DataFrame is a DatetimeIndex indicating
        the time period the power flow analysis was conducted for; columns
        of the DataFrame are the edges as well as stations of the network
        topology.

        Parameters
        ----------
        pypsa : :pandas:`pandas.DataFrame<dataframe>`
            Results time series of active power P in kW from the
            `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_

            Provide this if you want to set values. For retrieval of data do
            not pass an argument

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Active power results from power flow analysis

        """
        return self._pfa_p

    @pfa_p.setter
    def pfa_p(self, pypsa):
        self._pfa_p = pypsa

    @property
    def pfa_q(self):
        """
        Reactive power results from power flow analysis in kvar.

        Holds power flow analysis results for reactive power for the last
        iteration step. Index of the DataFrame is a DatetimeIndex indicating
        the time period the power flow analysis was conducted for; columns
        of the DataFrame are the edges as well as stations of the network
        topology.

        Parameters
        ----------
        pypsa : :pandas:`pandas.DataFrame<dataframe>`
            Results time series of reactive power Q in kvar from the
            `PyPSA network <https://www.pypsa.org/doc/components.html#network>`_

            Provide this if you want to set values. For retrieval of data do not
            pass an argument

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Reactive power results from power flow analysis

        """
        return self._pfa_q

    @pfa_q.setter
    def pfa_q(self, pypsa):
        self._pfa_q = pypsa

    @property
    def v_res(self):
        """
        Voltages at nodes in p.u. from last power flow analysis.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<frame>`
            Dataframe with voltages at nodes in p.u. from power flow analysis.
            Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the bus names of all buses in the analyzed grids.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<frame>`
            Dataframe with voltages at nodes in p.u. from power flow analysis.
            Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the bus names of all buses in the analyzed grids.

        """
        return self._v_res

    @v_res.setter
    def v_res(self, df):
        self._v_res = df

    @property
    def i_res(self):
        """
        Current results in kA from last power flow analysis.

        Parameters
        ----------
        df : :pandas:`pandas.DataFrame<frame>`
            Dataframe with currents in kA from power flow analysis.
            Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the line and transformer names of all lines and
            transformers in the analyzed grids.

            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<frame>`
            Dataframe with currents in kA from power flow analysis.
            Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the line and transformer names of all lines and
            transformers in the analyzed grids.

        """
        return self._i_res

    @i_res.setter
    def i_res(self, df):
        self._i_res = df

    @property
    def s_res(self):
        """
        Get resulting apparent power in MVA over lines and transformers.

        The apparent power returned is the highest apparent power determined
        from active and reactive power at the line endings / transformer
        sides.

        .. math::

            S = max(\sqrt{p_0^2 + q_0^2}, \sqrt{p_1^2 + q_1^2})

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Apparent power in MVA over lines and transformers.
            Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>` indicating the time
            steps the power flow analysis was conducted for; columns of the
            dataframe are the line and transformer names of all lines and
            transformers in the analyzed grids.

        """
        if self.pfa_p is None:
            return None

        return np.hypot(self.pfa_p, self.pfa_q)

    @property
    def equipment_changes(self):
        """
        Tracks changes in the equipment (e.g. replaced or added cable, etc.)

        The DataFrame is indexed by the component(
        :class:`~.network.components.Line`, :class:`~.network.components.Station`,
        etc.) and has the following columns:

        equipment : detailing what was changed (line, station, storage,
        curtailment). For ease of referencing we take the component itself.
        For lines we take the line-dict, for stations the transformers, for
        storage units the storage-object itself and for curtailment
        either a dict providing the details of curtailment or a curtailment
        object if this makes more sense (has to be defined).

        change : :obj:`str`
            Specifies if something was added or removed.

        iteration_step : :obj:`int`
            Used for the update of the pypsa network to only consider changes
            since the last power flow analysis.

        quantity : :obj:`int`
            Number of components added or removed. Only relevant for
            calculation of network expansion costs to keep track of how many
            new standard lines were added.

        Parameters
        ----------
        changes : :pandas:`pandas.DataFrame<dataframe>`
            Provide this if you want to set values. For retrieval of data do
            not pass an argument.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Equipment changes

        """
        return self._equipment_changes

    @equipment_changes.setter
    def equipment_changes(self, changes):
        self._equipment_changes = changes

    @property
    def grid_expansion_costs(self):
        """
        Holds network expansion costs in kEUR due to network expansion measures
        tracked in self.equipment_changes and calculated in
        edisgo.flex_opt.costs.grid_expansion_costs()

        Parameters
        ----------
        total_costs : :pandas:`pandas.DataFrame<dataframe>`

            DataFrame containing type and costs plus in the case of lines the
            line length and number of parallel lines of each reinforced
            transformer and line. Provide this if you want to set
            grid_expansion_costs. For retrieval of costs do not pass an
            argument.

            Index of the DataFrame is the respective object
            that can either be a :class:`~.network.components.Line` or a
            :class:`~.network.components.Transformer`. Columns are the following:

            type : :obj:`str`
                Transformer size or cable name

            total_costs : :obj:`float`
                Costs of equipment in kEUR. For lines the line length and
                number of parallel lines is already included in the total
                costs.

            quantity : :obj:`int`
                For transformers quantity is always one, for lines it specifies
                the number of parallel lines.

            line_length : :obj:`float`
                Length of line or in case of parallel lines all lines in km.

            voltage_level : :obj:`str`
                Specifies voltage level the equipment is in ('lv', 'mv' or
                'mv/lv').

            mv_feeder : :class:`~.network.components.Line`
                First line segment of half-ring used to identify in which
                feeder the network expansion was conducted in.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Costs of each reinforced equipment in kEUR.

        Notes
        -------
        Total network expansion costs can be obtained through
        costs.total_costs.sum().

        """
        return self._grid_expansion_costs

    @grid_expansion_costs.setter
    def grid_expansion_costs(self, total_costs):
        self._grid_expansion_costs = total_costs

    @property
    def grid_losses(self):
        """
        Holds active and reactive network losses in kW and kvar, respectively.

        Parameters
        ----------
        pypsa_grid_losses : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive network losses in columns 'p'
            and 'q' and in kW and kvar, respectively. Index is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive network losses in columns 'p'
            and 'q' and in kW and kvar, respectively. Index is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

        Notes
        ------
        Grid losses are calculated as follows:

        .. math::
            P_{loss} = \sum{feed-in} - \sum{load} + P_{slack}
            Q_{loss} = \sum{feed-in} - \sum{load} + Q_{slack}

        As the slack is placed at the secondary side of the HV/MV station
        losses do not include losses of the HV/MV transformers.

        """

        return self._grid_losses

    @grid_losses.setter
    def grid_losses(self, pypsa_grid_losses):
        self._grid_losses = pypsa_grid_losses

    @property
    def hv_mv_exchanges(self):
        """
        Holds active and reactive power exchanged with the HV network.

        The exchanges are essentially the slack results. As the slack is placed
        at the secondary side of the HV/MV station, this gives the energy
        transferred to and taken from the HV network at the secondary side of
        the HV/MV station.

        Parameters
        ----------
        hv_mv_exchanges : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive power exchanged with the HV
            network in columns 'p' and 'q' and in kW and kvar, respectively.
            Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`
            Dataframe holding active and reactive power exchanged with the HV
            network in columns 'p' and 'q' and in kW and kvar, respectively.
            Index is a :pandas:`pandas.DatetimeIndex<DatetimeIndex>`.

        """
        # ToDo: Instead of hv_mv_exchanges just use slack (is more general in
        # case only LV grid was analyzed)
        return self._hv_mv_exchanges

    @hv_mv_exchanges.setter
    def hv_mv_exchanges(self, hv_mv_exchanges):
        self._hv_mv_exchanges = hv_mv_exchanges

    @property
    def curtailment(self):
        """
        Holds curtailment assigned to each generator per curtailment target.

        ToDo: adapt to refactored code!

        Returns
        -------
        :obj:`dict` with :pandas:`pandas.DataFrame<dataframe>`
            Keys of the dictionary are generator types (and weather cell ID)
            curtailment targets were given for. E.g. if curtailment is provided
            as a :pandas:`pandas.DataFrame<dataframe>` with
            :pandas.`pandas.MultiIndex` columns with levels 'type' and
            'weather cell ID' the dictionary key is a tuple of
            ('type','weather_cell_id').
            Values of the dictionary are dataframes with the curtailed power in
            kW per generator and time step. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns are the
            generators of type
            :class:`edisgo.network.components.GeneratorFluctuating`.

        """
        raise NotImplementedError
        if self._curtailment is not None:
            result_dict = {}
            for key, gen_list in self._curtailment.items():
                curtailment_df = pd.DataFrame()
                for gen in gen_list:
                    curtailment_df[gen] = gen.curtailment
                result_dict[key] = curtailment_df
            return result_dict
        else:
            return None

    @property
    def storage_units(self):
        """
        Gathers relevant storage results.

        ToDo: adapt to refactored code!

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing all storage units installed in the MV and
            LV grids. Index of the dataframe are the storage representatives,
            columns are the following:

            nominal_power : :obj:`float`
                Nominal power of the storage in kW.

            voltage_level : :obj:`str`
                Voltage level the storage is connected to. Can either be 'mv'
                or 'lv'.

        """
        raise NotImplementedError

        grids = [self.edisgo_object.topology.mv_grid] + list(
            self.edisgo_object.topology.mv_grid.lv_grids
        )
        storage_results = {}
        storage_results["storage_id"] = []
        storage_results["nominal_power"] = []
        storage_results["voltage_level"] = []
        storage_results["grid_connection_point"] = []
        for grid in grids:
            for storage in grid.graph.nodes_by_attribute("storage"):
                storage_results["storage_id"].append(repr(storage))
                storage_results["nominal_power"].append(storage.nominal_power)
                storage_results["voltage_level"].append(
                    "mv" if isinstance(grid, MVGrid) else "lv"
                )
                storage_results["grid_connection_point"].append(
                    list(grid.graph.neighbors(storage))[0]
                )

        return pd.DataFrame(storage_results).set_index("storage_id")

    def storage_units_timeseries(self):
        """
        Returns a dataframe with storage time series.

        ToDo: adapt to refactored code!

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing time series of all storage units installed in
            the MV network and LV grids. Index of the dataframe is a
            :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns are the
            storage representatives.

        """
        raise NotImplementedError
        storage_units_p = pd.DataFrame()
        storage_units_q = pd.DataFrame()
        grids = [self.edisgo_object.topology.mv_grid] + list(
            self.edisgo_object.topology.mv_grid.lv_grids
        )
        for grid in grids:
            for storage in grid.graph.nodes_by_attribute("storage"):
                ts = storage.timeseries
                storage_units_p[repr(storage)] = ts.p
                storage_units_q[repr(storage)] = ts.q

        return storage_units_p, storage_units_q

    @property
    def storage_units_costs_reduction(self):
        """
        Contains costs reduction due to storage integration.

        ToDo: adapt to refactored code!

        Parameters
        ----------
        costs_df : :pandas:`pandas.DataFrame<dataframe>`
            Dataframe containing network expansion costs in kEUR before and after
            storage integration in columns 'grid_expansion_costs_initial' and
            'grid_expansion_costs_with_storage_units', respectively. Index of
            the dataframe is the MV network id.

        Returns
        -------
        :pandas:`pandas.DataFrame<dataframe>`

            Dataframe containing network expansion costs in kEUR before and after
            storage integration in columns 'grid_expansion_costs_initial' and
            'grid_expansion_costs_with_storage_units', respectively. Index of
            the dataframe is the MV network id.

        """
        raise NotImplementedError
        return self._storage_units_costs_reduction

    @storage_units_costs_reduction.setter
    def storage_units_costs_reduction(self, costs_df):
        self._storage_units_costs_reduction = costs_df

    @property
    def unresolved_issues(self):
        """
        Holds lines and nodes where over-loading or over-voltage issues
        could not be solved in network reinforcement.

        In case over-loading or over-voltage issues could not be solved
        after maximum number of iterations, network reinforcement is not
        aborted but network expansion costs are still calculated and unresolved
        issues listed here.

        Parameters
        ----------
        issues : dict

            Dictionary of critical lines/stations with relative over-loading
            and critical nodes with voltage deviation in p.u.. Format:

            .. code-block:: python

                {crit_line_1: rel_overloading_1, ...,
                 crit_line_n: rel_overloading_n,
                 crit_node_1: v_mag_pu_node_1, ...,
                 crit_node_n: v_mag_pu_node_n}

            Provide this if you want to set unresolved_issues. For retrieval
            of unresolved issues do not pass an argument.

        Returns
        -------
        Dictionary
            Dictionary of critical lines/stations with relative over-loading
            and critical nodes with voltage deviation in p.u.

        """
        return self._unresolved_issues

    @unresolved_issues.setter
    def unresolved_issues(self, issues):
        self._unresolved_issues = issues

    def save(self, directory, parameters="all"):
        """
        Saves results to disk.

        ToDo: adapt to refactored code!

        Depending on which results are selected and if they exist, the
        following directories and files are created:

        * `powerflow_results` directory

          * `voltages_pu.csv`

            See :py:attr:`~v_res` for more information.
          * `currents.csv`

            See :func:`~i_res` for more information.
          * `active_powers.csv`

            See :py:attr:`~pfa_p` for more information.
          * `reactive_powers.csv`

            See :py:attr:`~pfa_q` for more information.
          * `apparent_powers.csv`

            See :func:`~s_res` for more information.
          * `grid_losses.csv`

            See :py:attr:`~grid_losses` for more information.
          * `hv_mv_exchanges.csv`

            See :py:attr:`~hv_mv_exchanges` for more information.

        * `grid_expansion_results` directory

          * `grid_expansion_costs.csv`

            See :py:attr:`~grid_expansion_costs` for more information.
          * `equipment_changes.csv`

            See :py:attr:`~equipment_changes` for more information.
          * `unresolved_issues.csv`

            See :py:attr:`~unresolved_issues` for more information.

        * `curtailment_results` directory

          Files depend on curtailment specifications. There will be one file
          for each curtailment specification, that is for every key in
          :py:attr:`~curtailment` dictionary.

        * `storage_integration_results` directory

          * `storage_units.csv`

            See :func:`~storage_units` for more information.

        Parameters
        ----------
        directory : :obj:`str`
            Directory to save the results in.
        parameters : :obj:`str` or :obj:`list` of :obj:`str`
            Specifies which results will be saved. By default all results are
            saved. To only save certain results set `parameters` to one of the
            following options or choose several options by providing a list:

            * 'powerflow_results'
            * 'grid_expansion_results'
            * 'curtailment_results'
            * 'storage_integration_results'

        """

        def _save_power_flow_results(target_dir):
            if self.v_res is not None:
                # create directory
                os.makedirs(target_dir, exist_ok=True)

                # voltage
                self.v_res.to_csv(
                    os.path.join(target_dir, "voltages_pu.csv")
                )

                # current
                self.i_res.to_csv(os.path.join(target_dir, "currents.csv"))

                # active power
                self.pfa_p.to_csv(
                    os.path.join(target_dir, "active_powers.csv")
                )

                # reactive power
                self.pfa_q.to_csv(
                    os.path.join(target_dir, "reactive_powers.csv")
                )

                # apparent power
                self.s_res.to_csv(
                    os.path.join(target_dir, "apparent_powers.csv")
                )

                # network losses
                self.grid_losses.to_csv(
                    os.path.join(target_dir, "grid_losses.csv")
                )

                # network exchanges
                self.hv_mv_exchanges.to_csv(
                    os.path.join(target_dir, "hv_mv_exchanges.csv")
                )

        def _save_grid_expansion_results(target_dir):
            if self.grid_expansion_costs is not None:
                # create directory
                os.makedirs(target_dir, exist_ok=True)

                # network expansion costs
                self.grid_expansion_costs.to_csv(
                    os.path.join(target_dir, "grid_expansion_costs.csv")
                )

                # unresolved issues
                pd.DataFrame(self.unresolved_issues).to_csv(
                    os.path.join(target_dir, "unresolved_issues.csv")
                )

                # equipment changes
                self.equipment_changes.to_csv(
                    os.path.join(target_dir, "equipment_changes.csv")
                )

        def _save_curtailment_results(target_dir):
            pass
            # if self.curtailment is not None:
            #     # create directory
            #     os.makedirs(target_dir, exist_ok=True)
            #
            #     for key, curtailment_df in self.curtailment.items():
            #         if type(key) == tuple:
            #             type_prefix = '-'.join([key[0], str(key[1])])
            #         elif type(key) == str:
            #             type_prefix = key
            #         else:
            #             raise KeyError("Unknown key type {} for key {}".format(
            #                 type(key), key))
            #
            #         filename = os.path.join(
            #             target_dir, '{}.csv'.format(type_prefix))
            #
            #         curtailment_df.to_csv(filename, index_label=type_prefix)

        def _save_storage_integration_results(target_dir):
            pass
            # storages = self.storages
            # if not storages.empty:
            #     # create directory
            #     os.makedirs(target_dir, exist_ok=True)
            #
            #     # general storage information
            #     storages.to_csv(os.path.join(target_dir, 'storages.csv'))
            #
            #     # storages time series
            #     ts_p, ts_q = self.storages_timeseries()
            #     ts_p.to_csv(os.path.join(
            #         target_dir, 'storages_active_power.csv'))
            #     ts_q.to_csv(os.path.join(
            #         target_dir, 'storages_reactive_power.csv'))
            #
            #     if not self.storages_costs_reduction is None:
            #         self.storages_costs_reduction.to_csv(
            #             os.path.join(target_dir,
            #                          'storages_costs_reduction.csv'))

        # dictionary with function to call to save each parameter
        func_dict = {
            "powerflow_results": _save_power_flow_results,
            "grid_expansion_results": _save_grid_expansion_results,
            "curtailment_results": _save_curtailment_results,
            "storage_integration_results": _save_storage_integration_results,
        }

        # if string is given convert to list
        if isinstance(parameters, str):
            if parameters == "all":
                parameters = [
                    "powerflow_results",
                    "grid_expansion_results",
                    "curtailment_results",
                    "storage_integration_results",
                ]
            else:
                parameters = [parameters]

        # save each parameter
        for parameter in parameters:
            try:
                func_dict[parameter](os.path.join(directory, parameter))
            except KeyError:
                message = (
                    "Invalid input {} for `parameters` when saving "
                    "results. Must be any or a list of the following: "
                    "'powerflow_results', "
                    "'grid_expansion_results', 'curtailment_results', "
                    "'storage_integration_results'.".format(parameter)
                )
                logger.error(message)
                raise KeyError(message)
            except:
                raise
        # save measures
        pd.DataFrame(data={"measure": self.measures}).to_csv(
            os.path.join(directory, "measures.csv")
        )
        # save configs
        with open(os.path.join(directory, "configs.csv"), "w") as f:
            writer = csv.writer(f)
            rows = [
                ["{}".format(key)]
                + [value for item in values.items() for value in item]
                for key, values in self.edisgo_object.config._data.items()
            ]
            writer.writerows(rows)

    def from_csv(self, results_path, parameters):
        # measures
        measures_df = pd.read_csv(os.path.join(results_path, 'measures.csv'),
                                  index_col=0)
        self.measures = list(measures_df.measure.values)

        # if string is given convert to list
        if isinstance(parameters, str):
            if parameters == 'all':
                parameters = ['powerflow_results', 'grid_expansion_results',
                              'curtailment_results',
                              'storage_integration_results']
            else:
                parameters = [parameters]

        # import power flow results
        if 'powerflow_results' in parameters and os.path.isdir(os.path.join(
                results_path, 'powerflow_results')):
            # line loading
            self.i_res = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'currents.csv'),
                index_col=0, parse_dates=True)
            # voltage
            self.pfa_v_mag_pu = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'voltages_pu.csv'),
                index_col=0, parse_dates=True, header=[0, 1])
            # active power
            self.pfa_p = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'active_powers.csv'),
                index_col=0, parse_dates=True)
            # reactive power
            self.pfa_q = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'reactive_powers.csv'),
                index_col=0, parse_dates=True)
            # apparent power
            self.apparent_power = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'apparent_powers.csv'),
                index_col=0, parse_dates=True)
            # network losses
            self.grid_losses = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'grid_losses.csv'),
                index_col=0, parse_dates=True)
            # network exchanges
            self.hv_mv_exchanges = pd.read_csv(
                os.path.join(
                    results_path, 'powerflow_results', 'hv_mv_exchanges.csv'),
                index_col=0, parse_dates=True)
        else:
            self.i_res = None
            self.pfa_v_mag_pu = None
            self.pfa_p = None
            self.pfa_q = None
            self.apparent_power = None
            self.grid_losses = None
            self.hv_mv_exchanges = None

        # import network expansion results
        if 'grid_expansion_results' in parameters and os.path.isdir(
                os.path.join(results_path, 'grid_expansion_results')):
            # network expansion costs
            self.grid_expansion_costs = pd.read_csv(
                os.path.join(
                    results_path, 'grid_expansion_results',
                    'grid_expansion_costs.csv'),
                index_col=0)
            # equipment changes
            self.equipment_changes = pd.read_csv(
                os.path.join(
                    results_path, 'grid_expansion_results',
                    'equipment_changes.csv'),
                index_col=0)
        else:
            self.grid_expansion_costs = None
            self.equipment_changes = None

        # # import curtailment results
        # if 'curtailment_results' in parameters and os.path.isdir(
        #         os.path.join(results_path, 'curtailment_results')):
        #     self.curtailment = {}
        #     for file in os.listdir(os.path.join(
        #             results_path, 'curtailment_results')):
        #         if file.endswith(".csv"):
        #             try:
        #                 key = file[0:-4]
        #                 if '-' in key:
        #                     # make tuple if curtailment was given for generator
        #                     # type and weather cell id
        #                     tmp = key.split('-')
        #                     key = (tmp[0], float(tmp[1]))
        #                 self.curtailment[key] = pd.read_csv(
        #                     os.path.join(
        #                         results_path, 'curtailment_results', file),
        #                     index_col=0, parse_dates=True)
        #             except Exception as e:
        #                 logging.warning(
        #                     'The following error occured when trying to '
        #                     'import curtailment results: {}'.format(e))
        # else:
        #     self.curtailment = None

        # # import storage results
        # if 'storage_integration_results' in parameters and os.path.isdir(
        #         os.path.join(results_path, 'storage_integration_results')):
        #     # storages
        #     self.storages = pd.read_csv(
        #         os.path.join(results_path, 'storage_integration_results',
        #                      'storages.csv'),
        #         index_col=0)
        #     # storages costs reduction
        #     try:
        #         self.storages_costs_reduction = pd.read_csv(
        #             os.path.join(
        #                 results_path, 'storage_integration_results',
        #                 'storages_costs_reduction.csv'),
        #             index_col=0)
        #     except:
        #         pass
        #     # storages time series
        #     self.storages_p = pd.read_csv(
        #         os.path.join(
        #             results_path, 'storage_integration_results',
        #             'storages_active_power.csv'),
        #         index_col=0, parse_dates=True)
        #     # storages time series
        #     self.storages_q = pd.read_csv(
        #         os.path.join(
        #             results_path, 'storage_integration_results',
        #             'storages_reactive_power.csv'),
        #         index_col=0, parse_dates=True)
        #
        # else:
        #     self.storages = None
        #     self.storages_costs_reduction = None
        #     self.storages_p = None
        #     self.storages_q = None



class ResultsReimport:
    """
    Results class created from saved results.

    """
    def __init__(self, results_path, parameters='all'):
        raise NotImplementedError


#
#     def v_res(self, nodes=None, level=None):
#         """
#         Get resulting voltage level at node.
#
#         Parameters
#         ----------
#         nodes : :obj:`list`
#             List of string representatives of network topology components, e.g.
#             :class:`~.network.components.Generator`. If not provided defaults to
#             all nodes available in network level `level`.
#         level : :obj:`str`
#             Either 'mv' or 'lv' or None (default). Depending on which network
#             level results you are interested in. It is required to provide this
#             argument in order to distinguish voltage levels at primary and
#             secondary side of the transformer/LV station.
#             If not provided (respectively None) defaults to ['mv', 'lv'].
#
#         Returns
#         -------
#         :pandas:`pandas.DataFrame<dataframe>`
#             Resulting voltage levels obtained from power flow analysis
#
#         """
#         # check if voltages are available:
#         if hasattr(self, 'pfa_v_mag_pu'):
#             self.pfa_v_mag_pu.sort_index(axis=1, inplace=True)
#         else:
#             message = "No voltage results available."
#             raise AttributeError(message)
#
#         if level is None:
#             level = ['mv', 'lv']
#
#         if nodes is None:
#             return self.pfa_v_mag_pu.loc[:, (level, slice(None))]
#         else:
#             not_included = [_ for _ in nodes
#                             if _ not in list(self.pfa_v_mag_pu[level].columns)]
#             labels_included = [_ for _ in nodes if _ not in not_included]
#
#             if not_included:
#                 logging.warning("Voltage levels for {nodes} are not returned "
#                                 "from PFA".format(nodes=not_included))
#             return self.pfa_v_mag_pu[level][labels_included]
#
#     def s_res(self, components=None):
#         """
#         Get apparent power in kVA at line(s) and transformer(s).
#
#         Parameters
#         ----------
#         components : :obj:`list`
#             List of string representatives of :class:`~.network.components.Line`
#             or :class:`~.network.components.Transformer`. If not provided defaults
#             to return apparent power of all lines and transformers in the network.
#
#         Returns
#         -------
#         :pandas:`pandas.DataFrame<dataframe>`
#             Apparent power in kVA for lines and/or transformers.
#
#         """
#         if components is None:
#             return self.apparent_power
#         else:
#             not_included = [_ for _ in components
#                             if _ not in self.apparent_power.index]
#             labels_included = [_ for _ in components if _ not in not_included]
#
#             if not_included:
#                 logging.warning(
#                     "No apparent power results available for: {}".format(
#                         not_included))
#             return self.apparent_power.loc[:, labels_included]
#
#     def storages_timeseries(self):
#         """
#         Returns a dataframe with storage time series.
#
#         Returns
#         -------
#         :pandas:`pandas.DataFrame<dataframe>`
#
#             Dataframe containing time series of all storages installed in the
#             MV network and LV grids. Index of the dataframe is a
#             :pandas:`pandas.DatetimeIndex<DatetimeIndex>`. Columns are the
#             storage representatives.
#
#         """
#         return self.storages_p, self.storages_q
