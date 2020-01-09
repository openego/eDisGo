def opf_settings():
    opf_settings = {
        # name of postmethod
        "scenario": "nep",
        # objective function
        "objective": "nep",
        # chosen relaxation
        "relaxation": "none",
        # upper bound on network expansion
        "max_exp": 10,
        # number of time steps considered in optimization
        "time_horizon": 2,
        # length of time step in hours
        "time_elapsed": 1.0,
        # storage units are considered
        "storage_units": False,
        # positioning of storage units, if empty list, all buses are potential positions of storage units and
        # capacity is optimized
        "storage_buses": [],
        # total storage capacity in the network
        "total_storage_capacity": 0.0,
        # Requirements for curtailment in every time step is considered
        "curtailment_requirement": False,
        # List of total curtailment for each time step, len(list)== "time_horizon"
        "curtailment_requirement_series": [],
        # An overall allowance of curtailment is considered
        "curtailment_allowance": False,
        # Maximal allowed curtailment over entire time horizon,
        # DEFAULT: "3percent"=> 3% of total RES generation in time horizon may be curtailed, else: Float
        "curtailment_total": "3percent",
        }
    return opf_settings
