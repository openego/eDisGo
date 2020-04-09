def opf_settings():
    opf_settings = {
        # name of postmethod, right now this is just the scenario name and every scenario is handle in one problem setup
        # the so-called postmethod
        # for future extension define multiple postmethods in different julia files which will be call by the scenarioname
        "scenario": "nep",
        # objective function, DEFAULT "nep", future extension might include "generation costs" "storage costs" etc.
        "objective": "nep",
        # chosen relaxation, DEFAULT: "none", options: "none", "soc", "soc_cr", "cr", relaxation are described in
        # masterthesis "MULTIPERIOD OPTIMAL POWER FLOW PROBLEM IN DISTRIBUTION SYSTEM PLANNING" by Jaap Pedersen
        "relaxation": "none",
        # Dictionary of linked time steps in the format {linked step => original step}
        "clusters": {},
        # upper bound on network expansion, int, DEFAULT: 10
        "max_exp": 10,
        # number of time steps considered in optimization
        "time_horizon": 2,
        # length of time step in hours
        "time_elapsed": 1.0,
        # storage units are considered, DEFAULT:False, if true storage units will be located either at buses given in
        # "storage_buses" or if "storage_buses"=[] all buses are considered as possible locations
        "storage_units": False,
        # positioning of storage units, if empty list, all buses are potential positions of storage units and
        # capacity is optimized, entries of list need to be type "int"
        "storage_buses": [],
        # Only optimize operation of storages exported from eDisGo. Do not optimize storage positioning.
        "storage_operation_only": False,
        # total storage capacity in the network, sizing of storages is a decision variable and will be found in optimization
        "total_storage_capacity": 0.0,
        # Requirements for curtailment in every time step is considered, DEFAULT: False
        "curtailment_requirement": False,
        # List of total curtailment for each time step, len(list)== "time_horizon"
        "curtailment_requirement_series": [],
        # An overall allowance of curtailment is considered DEFAULT: False,
        "curtailment_allowance": False,
        # Maximal allowed curtailment over entire time horizon,
        # DEFAULT: 0.0, float
        "curtailment_total": 0.0,
        # Solver options
        # DEFAULT: IPOPT, right now only option
        "solver": "Ipopt",
        "solver_tol": 1e-8,
        # estimated working space for MUMPS if IPOPT,
        # smaller values might reduce required memory requirements, DEFAULT: 1000
        # see: https://coin-or.github.io/Ipopt/OPTIONS.html
        "mumps_mem_percent": 1000,
    }
    return opf_settings
