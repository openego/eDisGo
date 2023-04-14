from __future__ import annotations

import logging
import random

from typing import TYPE_CHECKING

import pandas as pd
import saio

from sqlalchemy.engine.base import Engine

from edisgo.io.db import session_scope_egon_data
from edisgo.tools.tools import (
    determine_bus_voltage_level,
    determine_grid_integration_voltage_level,
)

if TYPE_CHECKING:
    from edisgo import EDisGo

logger = logging.getLogger(__name__)


def home_batteries_oedb(
    edisgo_obj: EDisGo,
    scenario: str,
    engine: Engine,
):
    """
    Gets home battery data from oedb and integrates them into the grid.

    See :attr:`~.edisgo.EDisGo.import_home_batteries` for more information.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    scenario : str
        Scenario for which to retrieve home battery data. Possible options
        are "eGon2035" and "eGon100RE".
    engine : :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine.

    Returns
    --------
    list(str)
        List with names (as in index of
        :attr:`~.network.topology.Topology.storage_units_df`) of integrated storage
        units.

    """
    saio.register_schema("supply", engine)
    from saio.supply import egon_home_batteries

    with session_scope_egon_data(engine) as session:
        query = (
            session.query(
                egon_home_batteries.building_id,
                egon_home_batteries.p_nom,
                egon_home_batteries.capacity,
            )
            .filter(
                egon_home_batteries.scenario == scenario,
                egon_home_batteries.building_id.in_(
                    edisgo_obj.topology.loads_df.building_id.unique()
                ),
                egon_home_batteries.p_nom
                <= edisgo_obj.config["grid_connection"]["upper_limit_voltage_level_4"],
            )
            .order_by(egon_home_batteries.index)
        )
        batteries_df = pd.read_sql(sql=query.statement, con=engine, index_col=None)

    return _home_batteries_grid_integration(edisgo_obj, batteries_df)


def _home_batteries_grid_integration(edisgo_obj, batteries_df):
    """
    Integrates home batteries into the grid.

    See :attr:`~.edisgo.EDisGo.import_home_batteries` for more information on grid
    integration.

    Parameters
    ----------
    edisgo_obj : :class:`~.EDisGo`
    batteries_df : :pandas:`pandas.DataFrame<DataFrame>`
        Dataframe containing data on home storage units to integrate into the grid.
        Columns are:

            * p_nom : float
                Nominal electric power of storage in MW.
            * building_id : int
                Building ID of the building the storage is in.
            * capacity : float
                Storage capacity in MWh.

    Returns
    --------
    list(str)
        List with names (as in index of
        :attr:`~.network.topology.Topology.storage_units_df`) of integrated storage
        units.

    """

    def _integrate(bat_df):
        # filter batteries that are too large to be integrated into LV level
        batteries_large = bat_df[
            bat_df.p_nom
            > edisgo_obj.config["grid_connection"]["upper_limit_voltage_level_7"]
        ]
        batteries_small = bat_df[
            bat_df.p_nom
            <= edisgo_obj.config["grid_connection"]["upper_limit_voltage_level_7"]
        ]

        # integrate small batteries at buildings
        edisgo_obj.topology.storage_units_df = pd.concat(
            [edisgo_obj.topology.storage_units_df, batteries_small]
        )
        int_bats = batteries_small.index

        # integrate larger batteries - if generator/load is already connected to
        # higher voltage level it can be integrated at same bus, otherwise it is
        # integrated based on geolocation
        int_bats_own_grid_conn = pd.Index([])
        for bat in batteries_large.index:
            # check if building is already connected to a voltage level equal to or
            # higher than the voltage level the battery should be connected to
            bus = batteries_large.at[bat, "bus"]
            voltage_level_bus = determine_bus_voltage_level(edisgo_obj, bus)
            voltage_level_bat = determine_grid_integration_voltage_level(
                edisgo_obj, batteries_large.at[bat, "p_nom"]
            )

            if voltage_level_bat >= voltage_level_bus:
                # integrate at same bus as generator/load
                edisgo_obj.topology.storage_units_df = pd.concat(
                    [
                        edisgo_obj.topology.storage_units_df,
                        batteries_large.loc[[bat], :],
                    ]
                )
                int_bats = int_bats.append(pd.Index([bat]))
            else:
                # integrate based on geolocation
                bat_name = edisgo_obj.integrate_component_based_on_geolocation(
                    comp_type="storage_unit",
                    voltage_level=voltage_level_bat,
                    geolocation=(
                        edisgo_obj.topology.buses_df.at[bus, "x"],
                        edisgo_obj.topology.buses_df.at[bus, "y"],
                    ),
                    add_ts=False,
                    p_nom=batteries_large.at[bat, "p_nom"],
                    max_hours=batteries_large.at[bat, "max_hours"],
                    building_id=batteries_large.at[bat, "building_id"],
                    type="home_storage",
                )
                int_bats = int_bats.append(pd.Index([bat_name]))
                int_bats_own_grid_conn = int_bats_own_grid_conn.append(pd.Index([bat]))
        return int_bats, int_bats_own_grid_conn

    # add further information needed in storage_units_df
    batteries_df["max_hours"] = batteries_df["capacity"] / batteries_df["p_nom"]
    batteries_df.drop("capacity", axis=1, inplace=True)
    batteries_df["type"] = "home_storage"
    batteries_df["control"] = "PQ"
    # add storage name as index
    batteries_df["index"] = batteries_df.apply(
        lambda _: f"Storage_{_.building_id}", axis=1
    )
    batteries_df.set_index("index", drop=True, inplace=True)

    # check for duplicated storage names and choose random name for duplicates
    tmp = batteries_df.index.append(edisgo_obj.topology.storage_units_df.index)
    duplicated_indices = tmp[tmp.duplicated()]
    for duplicate in duplicated_indices:
        # find unique name
        random.seed(a=duplicate)
        new_name = duplicate
        while new_name in tmp:
            new_name = f"{duplicate}_{random.randint(10 ** 1, 10 ** 2)}"
        # change name in batteries_df
        batteries_df.rename(index={duplicate: new_name}, inplace=True)

    # integrate into grid
    # first try integrating at same bus as PV rooftop plant
    if "building_id" in edisgo_obj.topology.generators_df.columns:
        # join bus information for those storage units that are in the same building
        # as a generator
        generators_df = edisgo_obj.topology.generators_df
        building_id_busses = (
            generators_df.drop_duplicates(subset=["building_id"])
            .set_index("building_id")
            .loc[:, ["bus"]]
        )
        batteries_df = batteries_df.join(
            building_id_busses, how="left", on="building_id"
        )

        # differentiate between batteries that can be integrated using generator bus ID
        # and those using load bus ID
        batteries_gens_df = batteries_df.dropna(subset=["bus"])
        batteries_loads_df = batteries_df[batteries_df.bus.isna()]
        batteries_loads_df.drop("bus", axis=1, inplace=True)

        # integrate batteries that can be integrated at generator bus
        integrated_batteries, integrated_batteries_own_grid_conn = _integrate(
            batteries_gens_df
        )

    else:
        batteries_loads_df = batteries_df
        integrated_batteries = pd.Index([])
        integrated_batteries_own_grid_conn = pd.Index([])

    # integrate remaining home batteries at same bus as building
    if not batteries_loads_df.empty:
        # join busses corresponding to building ID
        loads_df = edisgo_obj.topology.loads_df
        building_id_busses = (
            loads_df[loads_df.type == "conventional_load"]
            .drop_duplicates(subset=["building_id"])
            .set_index("building_id")
            .loc[:, ["bus"]]
        )
        batteries_loads_df = batteries_loads_df.join(
            building_id_busses, how="left", on="building_id"
        )

        # integrate batteries that can be integrated at load bus
        integrated_batteries_2, integrated_batteries_own_grid_conn_2 = _integrate(
            batteries_loads_df
        )
        integrated_batteries = integrated_batteries.append(integrated_batteries_2)
        integrated_batteries_own_grid_conn = integrated_batteries_own_grid_conn.append(
            integrated_batteries_own_grid_conn_2
        )

    # check if all storage units were integrated
    if not len(batteries_df) == len(integrated_batteries):
        raise ValueError("Not all home batteries could be integrated into the grid.")

    # logging messages
    logger.debug(f"{sum(batteries_df.p_nom):.2f} MW of home batteries integrated.")
    if not batteries_loads_df.empty:
        logger.debug(
            f"Of this {sum(batteries_loads_df.p_nom):.2f} MW do not have a generator "
            f"with the same building ID."
        )
    if len(integrated_batteries_own_grid_conn) > 0:
        logger.debug(
            f"{sum(batteries_df.loc[integrated_batteries_own_grid_conn, 'p_nom']):.2f} "
            f"MW of home battery capacity was integrated at a new bus."
        )

    return integrated_batteries
