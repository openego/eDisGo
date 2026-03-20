"""
Helper functions to build scenario filters that replicate mview logic.

These functions create SQLAlchemy filter conditions that match the original
materialized view definitions, allowing direct queries on base tables.

Author: Generated for mviews replacement
Date: 2025-12-17
"""


def build_conv_scenario_filter(orm_table, scenario, version=None, preversion="v0.3.0"):
    """
    Build filter conditions for conventional power plants matching mview logic.

    Parameters
    ----------
    orm_table : SQLAlchemy Table
        The ORM table object (e.g., ego_dp_conv_powerplant)
    scenario : str
        Scenario name ('NEP 2035', 'eGo 100', 'Status Quo')
    version : str or list, optional
        Version filter. If None, uses defaults based on scenario.
    preversion : str, default 'v0.3.0'
        Preversion filter

    Returns
    -------
    SQLAlchemy filter condition
        Combined filter matching the mview definition
    """
    # Set default versions
    if version is None:
        if scenario in ["NEP 2035", "eGo 100"]:
            versions = ["v0.4.2", "v0.4.4", "v0.4.5"]
        else:
            versions = ["v0.4.5"]
    else:
        versions = [version] if isinstance(version, str) else version

    # Base filters
    filters = [
        orm_table.capacity > 0,
        orm_table.preversion == preversion,
    ]

    # Version filter
    if hasattr(orm_table, "version"):
        filters.append(orm_table.version.in_(versions))

    # Scenario-specific filters
    if scenario == "eGo 100":
        # eGo 100: only pumped storage from NEP 2035
        filters.extend(
            [
                orm_table.scenario == "NEP 2035",
                orm_table.fuel == "pumped_storage",
                (orm_table.shutdown == None) | (orm_table.shutdown >= 2049),
            ]
        )
    elif scenario == "NEP 2035":
        # NEP 2035: exclude hydro, filter by shutdown
        # Note: OEP doesn't support NOT IN, so we use individual != filters
        filters.extend(
            [
                orm_table.scenario == "NEP 2035",
                orm_table.fuel != "hydro",
                orm_table.fuel != "run_of_river",
                orm_table.fuel != "reservoir",
                (orm_table.shutdown == None) | (orm_table.shutdown >= 2034),
            ]
        )
    else:
        # Generic scenario filter
        filters.append(orm_table.scenario == scenario)

    # Combine all filters with AND
    from sqlalchemy import and_

    return and_(*filters)


def build_res_scenario_filter(orm_table, scenario, version=None, preversion="v0.3.0"):
    """
    Build filter conditions for renewable power plants matching mview logic.

    IMPORTANT: NEP 2035 mview is a UNION of Status Quo + NEP 2035 generators!
    The mview definition shows that ego_dp_res_powerplant_nep2035_mview contains
    BOTH Status Quo (solar/wind only, excl. offshore) AND NEP 2035 generators.

    Parameters
    ----------
    orm_table : SQLAlchemy Table
        The ORM table object (e.g., ego_dp_res_powerplant)
    scenario : str
        Scenario name ('NEP 2035', 'eGo 100', 'Status Quo')
    version : str or list, optional
        Version filter. If None, uses defaults based on scenario.
    preversion : str, default 'v0.3.0'
        Preversion filter

    Returns
    -------
    SQLAlchemy filter condition
        Combined filter matching the mview definition
    """
    from sqlalchemy import and_, or_

    # Set default versions
    if version is None:
        if scenario == "Status Quo":
            versions = ["v0.4.4", "v0.4.5"]
        else:
            # NEP 2035 uses v0.4.4 and v0.4.5
            versions = ["v0.4.4", "v0.4.5"]
    else:
        versions = [version] if isinstance(version, str) else version

    # Base filters (always required)
    base_filters = [
        orm_table.electrical_capacity > 0,
        orm_table.preversion == preversion,
    ]

    # Version filter
    if hasattr(orm_table, "version"):
        base_filters.append(orm_table.version.in_(versions))

    # Scenario-specific filters
    if scenario == "NEP 2035":
        # NEP 2035 mview is UNION of:
        # 1. Status Quo generators (solar/wind only, no offshore)
        # 2. NEP 2035 generators (all types)
        status_quo_filters = [
            orm_table.scenario == "Status Quo",
            orm_table.generation_type.in_(["solar", "wind"]),
            orm_table.generation_subtype != "wind_offshore",
        ]
        nep2035_filters = [orm_table.scenario == "NEP 2035"]

        # Combine: (Status Quo conditions) OR (NEP 2035 conditions)
        scenario_filter = or_(and_(*status_quo_filters), and_(*nep2035_filters))
        base_filters.append(scenario_filter)

    elif scenario == "Status Quo":
        # Status Quo: only solar and wind (excluding offshore)
        base_filters.extend(
            [
                orm_table.scenario == "Status Quo",
                orm_table.generation_type.in_(["solar", "wind"]),
                orm_table.generation_subtype != "wind_offshore",
            ]
        )
    else:
        # Generic scenario filter for other scenarios (like eGo 100)
        base_filters.append(orm_table.scenario == scenario)

    # Combine all filters with AND
    return and_(*base_filters)
