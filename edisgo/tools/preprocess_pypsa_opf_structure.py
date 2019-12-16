import numpy as np
import pandas as pd
from edisgo.flex_opt.costs import line_expansion_costs


def preprocess_pypsa_opf_structure(edisgo_grid, psa_network):
    """
    Prepare PyPsa Network for OPF Problem
    - add line costs
    - add hv side of HVMV-Transformer to network
    - move slack to hv side of HVMV-Transformer
    :param edisgo_grid: eDisGo obj
    :param psa_network: PyPsaNetwork
    :return:
    """
    mode = "mv"
    mv_grid = edisgo_grid.topology.mv_grid

    # add line expansion costs for all lines
    if not hasattr(psa_network.lines, "costs_earthworks") or psa_network.lines.costs_earthworks.dropna().empty:
        line_names = psa_network.lines.index
        linecosts_df = line_expansion_costs(edisgo_grid, line_names)
        psa_network.lines = psa_network.lines.join(linecosts_df)
    else:
        print("line costs are already set")

    # check if generator slack has a fluctuating variable set
    gen_slack_loc = psa_network.generators.control == "Slack"

    is_fluct = psa_network.generators.fluctuating.loc[gen_slack_loc][0]
    if is_fluct != is_fluct:
        print("value of fluctuating for slack generator is {}, it is changed to zero".format(is_fluct))
        psa_network.generators.fluctuating.loc[gen_slack_loc] = 0

    # add HVMV Trafo to pypsa network and move slack from MV to HV side

    # get dataframe containing hvmv trafos
    hvmv_trafos = mv_grid.transformers_df
    slack_bus_hv_name = hvmv_trafos.iloc[0].bus0
    if slack_bus_hv_name in psa_network.buses.index:
        print("HV side of transformer already in buses")
        return
    # get name of old slack bus
    if any(psa_network.buses.control == "Slack"):
        slack_bus_mv = psa_network.buses.loc[psa_network.buses.control == "Slack"]
    else:
        slack_bus_mv = psa_network.buses.loc[psa_network.generators.loc[gen_slack_loc].bus[0]]

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
    psa_network.add("Transformer", "Transformer_hvmv_{}".format(psa_network.name), type=trafo_type,
                    bus0=slack_bus_hv_name, bus1=slack_bus_mv.index)

    t = psa_network.transformers.iloc[0]

    # from pypsa apply_transformer_types(network)
    # https://github.com/PyPSA/PyPSA/blob/cd21756de30231408d42c1215df1d42254a0b714/pypsa/pf.py#L511
    # assume simple line model
    t["r"] = trafo["vscr"]/100.
    t["x"] = np.sqrt((trafo["vsc"]/100.)**2 - t["r"]**2)
    t["s_nom"] = sum(mv_grid.transformers_df.s_nom)
    for attr in ["r","x"]:
        t[attr] /= len(mv_grid.transformers_df)
    t["trafo_costs"] = edisgo_grid.config["costs_transformers"][mode]

    psa_network.transformers.iloc[0] = t

    # add new slack bus to dict buses_t
    for key, val in psa_network.buses_t.items():
        if len(val.columns) != 0:
            try:
                val.insert(0, slack_bus_hv_name, [1.0] * len(psa_network.snapshots))
            except ValueError as e:
                print("ValueError: {}".format(e))
        # print(len(val.columns))

    return

