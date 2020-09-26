import numpy as np
import pandas as pd
from edisgo.flex_opt.costs import line_expansion_costs
from pypsa.descriptors import Dict


def preprocess_pypsa_opf_structure(edisgo_grid, psa_network, hvmv_trafo=False):
    """
    Prepares pypsa network for OPF problem.

    * adds line costs
    * adds HV side of HV/MV transformer to network
    * moves slack to HV side of HV/MV transformer

    Parameters
    ----------
    edisgo_grid : :class:`~.edisgo.EDisGo`
    psa_network : :pypsa:`pypsa.Network<network>`
    hvmv_trafo : :obj:`Boolean`
        If True, HV side of HV/MV transformer is added to buses and Slack
        generator is moved to HV side.

    """
    mode = "mv"
    mv_grid = edisgo_grid.topology.mv_grid

    # add line expansion costs for all lines
    # ToDo annualize line costs!!
    if (
        not hasattr(psa_network.lines, "costs_cable")
        or psa_network.lines.costs_earthworks.dropna().empty
    ):
        line_names = psa_network.lines.index
        linecosts_df = line_expansion_costs(edisgo_grid, line_names)
        psa_network.lines = psa_network.lines.join(linecosts_df)
    else:
        print("line costs are already set")

    # check if generator slack has a fluctuating variable set
    gen_slack_loc = psa_network.generators.control == "Slack"
    psa_network.buses.control.loc[
        psa_network.generators.bus.loc[gen_slack_loc]
    ] = "Slack"
    is_fluct = psa_network.generators.fluctuating.loc[gen_slack_loc][0]
    # check for nan value
    if is_fluct != is_fluct:
        print(
            "value of fluctuating for slack generator is {}, it is changed to zero".format(
                is_fluct
            )
        )
        psa_network.generators.fluctuating.loc[gen_slack_loc] = False
        psa_network.generators.p_nom.loc[gen_slack_loc] = False

    if not hvmv_trafo:
        print("no hvmv trafo is added")
        return
    # add HVMV Trafo to pypsa network and move slack from MV to HV side

    # get dataframe containing hvmv trafos
    hvmv_trafos = mv_grid.transformers_df
    slack_bus_hv_name = hvmv_trafos.iloc[0].bus0
    if slack_bus_hv_name in psa_network.buses.index:
        print("HV side of transformer already in buses")
        return
    # get name of old slack bus
    if any(psa_network.buses.control == "Slack"):
        slack_bus_mv = psa_network.buses.loc[
            psa_network.buses.control == "Slack"
        ]
    else:
        slack_bus_mv = psa_network.buses.loc[
            psa_network.generators.loc[gen_slack_loc].bus[0]
        ]

    # get trafo type from pypsa
    trafo_type = hvmv_trafos.type.iloc[0]
    trafo = psa_network.transformer_types.T[trafo_type]

    slack_bus_hv = pd.DataFrame(slack_bus_mv.copy()).transpose()
    slack_bus_hv.index = [slack_bus_hv_name]
    slack_bus_hv.v_nom = trafo["v_nom_0"]
    slack_bus_hv.control = "Slack"

    buses_df = slack_bus_hv.append(psa_network.buses)
    psa_network.buses = buses_df

    # Move Generator_slack to new slack bus
    psa_network.generators.bus.loc[gen_slack_loc] = slack_bus_hv_name

    # Add Transformer to network
    psa_network.add(
        "Transformer",
        "Transformer_hvmv_{}".format(psa_network.name),
        type=trafo_type,
        bus0=hvmv_trafos.iloc[0].bus0,
        bus1=hvmv_trafos.iloc[0].bus1,
    )

    t = psa_network.transformers.iloc[0]

    # from pypsa apply_transformer_types(network)
    # https://github.com/PyPSA/PyPSA/blob/cd21756de30231408d42c1215df1d42254a0b714/pypsa/pf.py#L511
    # assume simple line model
    t["r"] = trafo["vscr"] / 100.0
    t["x"] = np.sqrt((trafo["vsc"] / 100.0) ** 2 - t["r"] ** 2)
    t["s_nom"] = sum(mv_grid.transformers_df.s_nom)
    for attr in ["r", "x"]:
        t[attr] /= len(mv_grid.transformers_df)

    psa_network.transformers.iloc[
        0
    ] = t.transpose()  # pd.DataFrame(t).transpose()

    psa_network.transformers["trafo_costs"] = edisgo_grid.config[
        "costs_transformers"
    ][mode]
    # print(psa_network.transformers)
    print(hasattr(psa_network.transformers, "trafo_costs"))
    # add new slack bus to dict buses_t
    for key, val in psa_network.buses_t.items():
        if len(val.columns) != 0:
            try:
                val.insert(
                    0, slack_bus_hv_name, [1.0] * len(psa_network.snapshots)
                )
            except ValueError as e:
                print("ValueError: {}".format(e))


def aggregate_fluct_generators(psa_network):
    """
    Aggregates fluctuating generators of same type at the same node.

    Iterates over all generator buses. If multiple fluctuating generators are
    attached, they are aggregated by type.

    Parameters
    ----------
    psa_network: :pypsa:`pypsa.Network<network>`

    """
    gen_df = psa_network.generators.copy()
    gen_t_dict = psa_network.generators_t.copy()
    gen_buses = np.unique(gen_df.bus)
    gen_aggr_df_all = pd.DataFrame(columns=gen_df.columns)
    for gen_bus in gen_buses:
        gens = gen_df[gen_df.bus == gen_bus]
        n_gens = len(gens)
        if n_gens <= 1:
            # no generators to aggregate at this bus
            continue
        else:
            print("{} has {} generators attached".format(gen_bus, n_gens))

            for fluct in ["wind", "solar"]:
                if "mvgd" in gen_bus:
                    gen_name = "Generator_aggr_{}_{}".format(
                        gen_bus[gen_bus.index("mvgd") :], fluct
                    )
                else:
                    gen_name = "Generator_aggr_{}_{}".format(gen_bus, fluct)
                # ToDo check for type rather than generator name
                gens_to_aggr = gens.loc[gens.index.str.contains(fluct)]
                print("{} gens of type {}".format(len(gens_to_aggr), fluct))
                if len(gens_to_aggr) == 0:
                    continue
                gen_aggr_df = pd.DataFrame(
                    {
                        "bus": [gens_to_aggr.bus[0]],
                        "control": ["PQ"],
                        "p_set": gens_to_aggr["p_set"].iloc[0],
                        "q_set": gens_to_aggr["q_set"].iloc[0],
                        "p_nom": [sum(gens_to_aggr.p_nom)],
                        "start_up_cost": gens_to_aggr["start_up_cost"].iloc[0],
                        "shut_down_cost": gens_to_aggr["shut_down_cost"].iloc[
                            0
                        ],
                        "marginal_cost": gens_to_aggr["marginal_cost"].iloc[0],
                        "fluctuating": [True],
                    },
                    index=[gen_name],
                )
                gen_aggr_df_all = gen_aggr_df_all.append(gen_aggr_df)
                # drop aggregated generators and add new generator to generator dataframe
                gen_df = gen_df.drop(gens_to_aggr.index)
                gen_df = gen_df.append(gen_aggr_df)
                # gens = gens.drop(gens_to_aggr.index)

                # sum timeseries for aggregated generators
                p_set_sum = pd.DataFrame(
                    gen_t_dict["p_set"][gens_to_aggr.index].agg("sum", axis=1),
                    columns=[gen_name],
                )
                gen_t_dict["p_set"] = gen_t_dict["p_set"].drop(
                    gens_to_aggr.index, axis=1
                )
                gen_t_dict["p_set"] = gen_t_dict["p_set"].join(p_set_sum)

                q_set_sum = pd.DataFrame(
                    gen_t_dict["q_set"][gens_to_aggr.index].agg("sum", axis=1),
                    columns=[gen_name],
                )
                gen_t_dict["q_set"] = gen_t_dict["q_set"].drop(
                    gens_to_aggr.index, axis=1
                )
                gen_t_dict["q_set"] = gen_t_dict["q_set"].join(q_set_sum)

    # print(gen_df.shape[0] == gen_t_dict["p_set"].shape[1] and gen_df.shape[0] ==
    #       gen_t_dict["q_set"].shape[1])

    # write aggregated generator dataframe on pypsa network
    psa_network.generators = gen_df
    # write aggregated timeseries into psa_network.generators_t as pypsa.descriptors.Dict()
    psa_network.generators_t = Dict(gen_t_dict)
