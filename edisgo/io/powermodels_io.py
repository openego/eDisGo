"""
This module provides tools to convert eDisGo representation of the network
topology and timeseries to PowerModels network data format . Call :func:`to_powermodels`
to retrieve the PowerModels network container.
"""

# import math

# import numpy as np
# import pandas as pd
import pypsa

from edisgo.io.pypsa_io import to_pypsa


def to_powermodels(edisgo_object):
    # convert eDisGo object to pypsa network structure
    psa_net = to_pypsa(edisgo_object)
    # calculate per unit values
    pypsa.pf.calculate_dependent_values(psa_net)

    # build static PowerModels structure
    pm = _init_pm()
    timehorizon = len(psa_net.snapshots)  # length of considered timesteps
    pm["name"] = "ding0_{}_t_{}".format(edisgo_object.topology.id, timehorizon)

    return pm


def _init_pm():
    # init empty powermodels dictionary
    pm = {
        "gen": dict(),
        "branch": dict(),
        "bus": dict(),
        "dcline": dict(),
        "load": dict(),
        "storage": dict(),
        "baseMVA": 1,
        "source_version": 2,
        "shunt": dict(),
        "sourcetype": "eDisGo",
        "per_unit": True,
        "name": "name",
    }
    return pm
