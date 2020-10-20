import json
import pandas as pd
import logging

from edisgo.tools.preprocess_pypsa_opf_structure import (
    preprocess_pypsa_opf_structure,
    aggregate_fluct_generators,
)

logger = logging.getLogger("edisgo")


def read_from_json(edisgo_obj, path, mode="mv"):
    """
    Read optimization results from json file.

    This reads the optimization results directly from a JSON result file
    without carrying out the optimization process.

    Parameters
    -----------
    edisgo_obj : :class:`~.EDisGo`
        An edisgo object with the same topology that the optimization was run
        on.
    path : str
        Path to the optimization result JSON file.
    mode : str
        Voltage level, currently only supports "mv"

    """
    pypsa_net = edisgo_obj.to_pypsa(
        mode=mode, timesteps=edisgo_obj.timeseries.timeindex
    )
    timehorizon = len(pypsa_net.snapshots)
    pypsa_net.name = "ding0_{}_t_{}".format(
        edisgo_obj.topology.id, timehorizon
    )
    preprocess_pypsa_opf_structure(edisgo_obj, pypsa_net, hvmv_trafo=False)
    aggregate_fluct_generators(pypsa_net)
    edisgo_obj.opf_results.set_solution(path, pypsa_net)


class LineVariables:
    def __init__(self):
        self.p = None
        self.q = None
        self.cm = None


class BusVariables:
    def __init__(self):
        self.w = None


class GeneratorVariables:
    def __init__(self):
        self.pg = None
        self.qg = None


class LoadVariables:
    def __init__(self):
        self.pd = None
        self.qd = None


class StorageVariables:
    def __init__(self):
        self.ud = None
        self.uc = None
        self.soc = None


class OPFResults:
    def __init__(self):
        self.solution_file = None
        self.solution_data = None
        self.name = None
        self.obj = None
        self.status = None
        self.solution_time = None
        self.solver = None
        self.lines = None
        self.pypsa = None
        self.lines_t = LineVariables()
        self.buses_t = BusVariables()
        self.generators_t = GeneratorVariables()
        self.loads_t = LoadVariables()
        self.storage_units = None
        self.storage_units_t = StorageVariables()

    def set_solution(self, solution_name, pypsa_net):
        self.read_solution_file(solution_name)
        self.set_solution_to_results(pypsa_net)

    def read_solution_file(self, solution_name):
        with open(solution_name) as json_file:
            self.solution_data = json.load(json_file)
        self.solution_file = solution_name

    def dump_solution_file(self, solution_name=[]):
        if not solution_name:
            solution_name = self.solution_name
        with open(solution_name, "w") as outfile:
            json.dump(self.solution_data, outfile)

    def set_solution_to_results(self, pypsa_net):
        solution_data = self.solution_data
        self.name = solution_data["name"]
        self.obj = solution_data["obj"]
        self.status = solution_data["status"]
        self.solution_time = solution_data["sol_time"]
        self.solver = solution_data["solver"]
        self.pypsa = pypsa_net
        # Line Variables
        self.set_line_variables(pypsa_net)
        # Bus Variables
        self.set_bus_variables(pypsa_net)
        # Generator Variables
        # TODO Adjust for case that generators are fixed and no variables are returned from julia
        self.set_gen_variables(pypsa_net)
        self.set_load_variables(pypsa_net)
        # Storage Variables
        self.set_strg_variables(pypsa_net)

    def set_line_variables(self, pypsa_net):
        solution_data = self.solution_data

        # time independent variables: ne: line expansion factor, 1.0 => no expansion
        br_statics = pd.Series(
            solution_data["branch"]["static"]["ne"], name="nep"
        ).to_frame()
        br_statics.index = br_statics.index.astype(int)
        br_statics = br_statics.sort_index()
        br_statics.index = pypsa_net.lines.index
        self.lines = br_statics

        # time dependent variables: cm: squared current magnitude, p: active power flow, q: reactive power flow
        ts = pypsa_net.snapshots.sort_values()
        cm_t = pd.DataFrame(index=ts, columns=pypsa_net.lines.index)
        p_t = pd.DataFrame(index=ts, columns=pypsa_net.lines.index)
        q_t = pd.DataFrame(index=ts, columns=pypsa_net.lines.index)
        for (t, date_idx) in enumerate(ts):
            branch_t = pd.DataFrame(solution_data["branch"]["nw"][str(t + 1)])
            branch_t.index = branch_t.index.astype(int)
            branch_t = branch_t.sort_index()
            branch_t.insert(0, "br_idx", branch_t.index)
            branch_t.index = pypsa_net.lines.index
            p_t.loc[date_idx] = branch_t.p.T
            q_t.loc[date_idx] = branch_t.q.T
            cm_t.loc[date_idx] = branch_t.cm.T
        self.lines_t.cm = cm_t
        self.lines_t.p = p_t
        self.lines_t.q = q_t
        return

    def set_bus_variables(self, pypsa_net):
        solution_data = self.solution_data
        ts = pypsa_net.snapshots.sort_values()
        w_t = pd.DataFrame(index=ts, columns=pypsa_net.buses.index)
        for (t, date_idx) in enumerate(ts):
            bus_t = pd.DataFrame(solution_data["bus"]["nw"][str(t + 1)])
            bus_t.index = bus_t.index.astype(int)
            bus_t = bus_t.sort_index()
            bus_t.index = pypsa_net.buses.index
            w_t.loc[date_idx] = bus_t.w
        self.buses_t.w = w_t
        return

    def set_gen_variables(self, pypsa_net):
        # ToDo disaggregate aggregated generators?
        gen_solution_data = self.solution_data["gen"]["nw"]
        ts = pypsa_net.snapshots.sort_values()

        # in case no curtailment requirement was given, all generators are
        # made to fixed loads and Slack is the only remaining generator
        if len(gen_solution_data["1"]["pg"].keys()) == 1:
            try:
                # check if generator index in pypsa corresponds to generator
                # int
                slack = pypsa_net.generators[
                    pypsa_net.generators.control == "Slack"
                ].index.values[0]
                ind = pypsa_net.generators.index.get_loc(slack) + 1
                if not str(ind) in gen_solution_data["1"]["pg"].keys():
                    logger.warning("Slack indexes do not match.")
                # write slack results
                pg_t = pd.DataFrame(index=ts, columns=[slack])
                qg_t = pd.DataFrame(index=ts, columns=[slack])
                for (t, date_idx) in enumerate(ts):
                    gen_t = pd.DataFrame(gen_solution_data[str(t + 1)])
                    gen_t.index = [slack]
                    pg_t.loc[date_idx] = gen_t.pg
                    qg_t.loc[date_idx] = gen_t.qg
                self.generators_t.pg = pg_t
                self.generators_t.qg = qg_t
            except:
                logger.warning(
                    "Error in writing OPF solutions for slack time " "series."
                )
        else:
            try:
                pg_t = pd.DataFrame(
                    index=ts, columns=pypsa_net.generators.index
                )
                qg_t = pd.DataFrame(
                    index=ts, columns=pypsa_net.generators.index
                )
                for (t, date_idx) in enumerate(ts):
                    gen_t = pd.DataFrame(gen_solution_data[str(t + 1)])
                    gen_t.index = gen_t.index.astype(int)
                    gen_t = gen_t.sort_index()
                    gen_t.index = pypsa_net.generators.index
                    pg_t.loc[date_idx] = gen_t.pg
                    qg_t.loc[date_idx] = gen_t.qg
                self.generators_t.pg = pg_t
                self.generators_t.qg = qg_t
            except:
                logger.warning(
                    "Error in writing OPF solutions for generator time "
                    "series."
                )

    def set_load_variables(self, pypsa_net):

        load_solution_data = self.solution_data["load"]["nw"]
        ts = pypsa_net.snapshots.sort_values()

        pd_t = pd.DataFrame(
            index=ts,
            columns=[int(_) for _ in load_solution_data["1"]["pd"].keys()]
        )
        qd_t = pd.DataFrame(
            index=ts,
            columns=[int(_) for _ in load_solution_data["1"]["pd"].keys()]
        )
        for (t, date_idx) in enumerate(ts):
            load_t = pd.DataFrame(load_solution_data[str(t + 1)])
            load_t.index = load_t.index.astype(int)
            pd_t.loc[date_idx] = load_t.pd
            qd_t.loc[date_idx] = load_t.qd

        load_buses = self.pypsa.loads.bus.unique()
        load_bus_df = pd.DataFrame(
            columns=["bus_loc"],
            index=[load_buses])
        for b in load_buses:
            load_bus_df.at[
                b, "bus_loc"] = self.pypsa.buses.index.get_loc(b)
        load_bus_df = load_bus_df.sort_values(by="bus_loc").reset_index()
        load_bus_df.index = load_bus_df.index + 1
        pd_t = pd_t.rename(columns=load_bus_df.level_0.to_dict())
        qd_t = qd_t.rename(columns=load_bus_df.level_0.to_dict())
        self.loads_t.pd = pd_t
        self.loads_t.qd = qd_t

    def set_strg_variables(self, pypsa_net):
        solution_data = self.solution_data

        # time independent values
        strg_statics = pd.DataFrame.from_dict(
            solution_data["storage"]["static"]["emax"], orient="index"
        )
        if not strg_statics.empty:
            strg_statics.columns = ["emax"]

            strg_statics.index = strg_statics.index.astype(int)
            strg_statics = strg_statics.sort_index()

            # Convert one-based storage indices back to string names
            idx_names = [
                pypsa_net.buses.index[i - 1] for i in strg_statics.index
            ]
            strg_statics.index = pd.Index(idx_names)

            self.storage_units = strg_statics

            # time dependent values
            ts = pypsa_net.snapshots
            uc_t = pd.DataFrame(index=ts, columns=strg_statics.index)
            ud_t = pd.DataFrame(index=ts, columns=strg_statics.index)
            soc_t = pd.DataFrame(index=ts, columns=strg_statics.index)

            for (t, date_idx) in enumerate(ts):
                strg_t = pd.DataFrame(
                    solution_data["storage"]["nw"][str(t + 1)]
                )
                strg_t.index = strg_t.index.astype(int)
                strg_t = strg_t.sort_index()
                strg_t.index = strg_statics.index

                uc_t.loc[date_idx].update(strg_t["uc"])
                ud_t.loc[date_idx].update(strg_t["ud"])
                soc_t.loc[date_idx].update(strg_t["soc"])

            self.storage_units_t.soc = soc_t
            self.storage_units_t.uc = uc_t
            self.storage_units_t.ud = ud_t
