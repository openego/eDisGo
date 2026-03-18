"""
Replacement functions for deprecated materialized views (mviews) in eDisGo/oedb.

This module provides functions to replace the deprecated mviews by directly querying
the underlying base tables with appropriate filters. The filter logic is extracted
from the original mview definitions in mviews_definitions.sql.

Author: Generated for mviews replacement
Date: 2025-12-17
"""

import pandas as pd
from sqlalchemy import text
from typing import Optional, Literal


class MviewsReplacement:
    """
    Replacement class for loading data from deprecated mviews.

    This class provides methods to query base tables directly instead of
    using materialized views, applying the same filter logic as the original
    mview definitions.
    """

    def __init__(self, engine):
        """
        Initialize the MviewsReplacement with a database engine.

        Parameters
        ----------
        engine : sqlalchemy.engine.Engine
            SQLAlchemy database engine for executing queries
        """
        self.engine = engine

    def get_conv_powerplant(
        self,
        scenario: str,
        version: Optional[str] = None,
        preversion: str = 'v0.3.0',
        schema: Literal['model_draft', 'data'] = 'data',
        table_name: str = 'ego_dp_conv_powerplant'
    ) -> pd.DataFrame:
        """
        Load conventional power plant data (replaces ego_dp_conv_powerplant_*_mview).

        This function replicates the logic from:
        - mviews.ego_dp_conv_powerplant_ego100_mview
        - mviews.ego_dp_conv_powerplant_nep2035_mview
        - mviews.ego_supply_conv_powerplant_ego100_mview
        - mviews.ego_supply_conv_powerplant_nep2035_mview

        Parameters
        ----------
        scenario : str
            Scenario name (e.g., 'NEP 2035', 'eGo 100', 'Status Quo')
        version : str, optional
            Version string(s). If None, uses default versions for the scenario.
        preversion : str, default 'v0.3.0'
            Preversion filter
        schema : str, default 'data'
            Database schema name ('data' or 'model_draft')
        table_name : str, default 'ego_dp_conv_powerplant'
            Base table name (ego_dp_conv_powerplant or ego_dp_supply_conv_powerplant)

        Returns
        -------
        pd.DataFrame
            Conventional power plant data matching the mview criteria
        """
        # Set default versions based on scenario
        if version is None:
            if scenario in ['NEP 2035', 'eGo 100']:
                version_list = ['v0.4.2', 'v0.4.4', 'v0.4.5']
            else:
                version_list = ['v0.4.5']
        else:
            version_list = [version] if isinstance(version, str) else version

        # Build version filter
        version_filter = "', '".join(version_list)

        # Base query - selecting all columns from the table
        if scenario == 'eGo 100':
            # Special case for eGo 100: only pumped storage from NEP 2035
            query = f"""
            SELECT DISTINCT
                version, preversion, id, bnetza_id, company, name, postcode, city,
                street, state, block, commissioned_original, commissioned, retrofit,
                shutdown, status, fuel, technology, type, eeg, chp, capacity,
                capacity_uba, chp_capacity_uba, efficiency_data, efficiency_estimate,
                network_node, voltage, network_operator, name_uba, lat, lon,
                'pumped storage for eGo 100' AS comment,
                geom, voltage_level, subst_id, otg_id, un_id, la_id,
                'eGo 100' AS scenario,
                'constantly' AS flag,
                nuts
            FROM {schema}.{table_name}
            WHERE scenario = 'NEP 2035'
                AND fuel = 'pumped_storage'
                AND capacity > 0
                AND (shutdown IS NULL OR shutdown >= 2049)
                AND preversion = '{preversion}'
                AND version IN ('{version_filter}')
            """
        elif scenario == 'NEP 2035':
            # NEP 2035: exclude hydro plants, filter by shutdown date
            query = f"""
            SELECT DISTINCT
                version, id, bnetza_id, company, name, postcode, city,
                street, state, block, commissioned_original, commissioned, retrofit,
                shutdown, status, fuel, technology, type, eeg, chp, capacity,
                capacity_uba, chp_capacity_uba, efficiency_data, efficiency_estimate,
                network_node, voltage, network_operator, name_uba, lat, lon, comment,
                geom, voltage_level, subst_id, otg_id, un_id, preversion, la_id,
                scenario, flag, nuts
            FROM {schema}.{table_name}
            WHERE scenario = '{scenario}'
                AND capacity > 0
                AND fuel NOT IN ('hydro', 'run_of_river', 'reservoir')
                AND (shutdown IS NULL OR shutdown >= 2034)
                AND preversion = '{preversion}'
                AND version IN ('{version_filter}')
            """
        else:
            # Generic query for other scenarios
            query = f"""
            SELECT DISTINCT
                version, preversion, id, bnetza_id, company, name, postcode, city,
                street, state, block, commissioned_original, commissioned, retrofit,
                shutdown, status, fuel, technology, type, eeg, chp, capacity,
                capacity_uba, chp_capacity_uba, efficiency_data, efficiency_estimate,
                network_node, voltage, network_operator, name_uba, lat, lon, comment,
                geom, voltage_level, subst_id, otg_id, un_id, la_id, scenario, flag, nuts
            FROM {schema}.{table_name}
            WHERE scenario = '{scenario}'
                AND capacity > 0
                AND preversion = '{preversion}'
            """
            if version_list:
                query += f" AND version IN ('{version_filter}')"

        return pd.read_sql(text(query), self.engine)

    def get_res_powerplant(
        self,
        scenario: str,
        version: Optional[str] = None,
        preversion: str = 'v0.3.0',
        schema: Literal['model_draft', 'data'] = 'data',
        table_name: str = 'ego_dp_res_powerplant'
    ) -> pd.DataFrame:
        """
        Load renewable power plant data (replaces ego_dp_res_powerplant_*_mview).

        This function replicates the logic from:
        - mviews.ego_dp_res_powerplant_ego100_mview
        - mviews.ego_dp_res_powerplant_nep2035_mview
        - mviews.ego_supply_res_powerplant_ego100_mview
        - mviews.ego_supply_res_powerplant_nep2035_mview

        Parameters
        ----------
        scenario : str
            Scenario name (e.g., 'NEP 2035', 'eGo 100', 'Status Quo')
        version : str, optional
            Version string(s). If None, uses default versions.
        preversion : str, default 'v0.3.0'
            Preversion filter
        schema : str, default 'data'
            Database schema name
        table_name : str, default 'ego_dp_res_powerplant'
            Base table name (ego_dp_res_powerplant or ego_dp_supply_res_powerplant)

        Returns
        -------
        pd.DataFrame
            Renewable power plant data matching the mview criteria
        """
        # Set default versions based on scenario
        if version is None:
            if scenario == 'Status Quo':
                version_list = ['v0.4.4', 'v0.4.5']
            else:
                version_list = ['v0.4.5']
        else:
            version_list = [version] if isinstance(version, str) else version

        version_filter = "', '".join(version_list)

        # Base query with duplicate filtering logic from the mview
        if table_name == 'ego_dp_res_powerplant':
            # For ego_dp_res_powerplant: complex duplicate filtering
            id_column = "id || version"
        else:
            # For ego_dp_supply_res_powerplant: simpler duplicate filtering
            id_column = "id"

        if scenario == 'Status Quo':
            # Status Quo scenario: only solar and wind (excluding offshore)
            query = f"""
            WITH filtered_data AS (
                SELECT DISTINCT ON ({id_column})
                    version, id, start_up_date, electrical_capacity,
                    generation_type, generation_subtype, thermal_capacity,
                    city, postcode, address, lon, lat, gps_accuracy,
                    validation, notification_reason, eeg_id, tso, tso_eic,
                    dso_id, dso, voltage_level_var, network_node, power_plant_id,
                    source, comment, geom, subst_id, otg_id, un_id, voltage_level,
                    la_id, mvlv_subst_id, rea_sort, rea_flag, rea_geom_line,
                    rea_geom_new, preversion, flag, scenario, nuts, w_id
                FROM {schema}.{table_name}
                WHERE scenario = '{scenario}'
                    AND preversion = '{preversion}'
                    AND version IN ('{version_filter}')
                    AND electrical_capacity > 0
                    AND generation_type IN ('solar', 'wind')
                    AND generation_subtype != 'wind_offshore'
                    AND NOT EXISTS (
                        SELECT 1
                        FROM {schema}.{table_name} t2
                        WHERE t2.{id_column} = {table_name}.{id_column}
                            AND t2.version IN ('{version_filter}')
                        GROUP BY t2.{id_column}
                        HAVING COUNT(*) > 1
                    )
                ORDER BY {id_column}
            )
            SELECT * FROM filtered_data
            """
        else:
            # Other scenarios
            query = f"""
            SELECT DISTINCT ON ({id_column})
                version, id, start_up_date, electrical_capacity,
                generation_type, generation_subtype, thermal_capacity,
                city, postcode, address, lon, lat, gps_accuracy,
                validation, notification_reason, eeg_id, tso, tso_eic,
                dso_id, dso, voltage_level_var, network_node, power_plant_id,
                source, comment, geom, subst_id, otg_id, un_id, voltage_level,
                la_id, mvlv_subst_id, rea_sort, rea_flag, rea_geom_line,
                rea_geom_new, preversion, flag, scenario, nuts, w_id
            FROM {schema}.{table_name}
            WHERE scenario = '{scenario}'
                AND preversion = '{preversion}'
                AND electrical_capacity > 0
            """
            if version_list:
                query += f" AND version IN ('{version_filter}')"
            query += f" ORDER BY {id_column}"

        return pd.read_sql(text(query), self.engine)

    def get_loadarea(
        self,
        version: str = 'v0.4.5',
        schema: Literal['model_draft', 'data'] = 'data',
        table_name: str = 'ego_dp_loadarea'
    ) -> pd.DataFrame:
        """
        Load load area data (replaces ego_dp_loadarea_*_mview).

        This function replicates the logic from:
        - mviews.ego_dp_loadarea_v0_4_3_mview
        - mviews.ego_dp_loadarea_v0_4_5_mview

        Parameters
        ----------
        version : str, default 'v0.4.5'
            Version string
        schema : str, default 'data'
            Database schema name
        table_name : str, default 'ego_dp_loadarea'
            Base table name

        Returns
        -------
        pd.DataFrame
            Load area data
        """
        query = f"""
        SELECT
            id, subst_id, area_ha, nuts, rs_0, ags_0, otg_id, un_id,
            zensus_sum, zensus_count, zensus_density, ioer_sum,
            ioer_count, ioer_density, sector_area_residential,
            sector_area_retail, sector_area_industrial,
            sector_area_agricultural, sector_share_residential,
            sector_share_retail, sector_share_industrial,
            sector_share_agricultural, sector_count_residential,
            sector_count_retail, sector_count_industrial,
            sector_count_agricultural, sector_consumption_residential,
            sector_consumption_retail, sector_consumption_industrial,
            sector_consumption_agricultural, sector_peakload_retail,
            sector_peakload_residential, sector_peakload_industrial,
            sector_peakload_agricultural, geom, geom_centre,
            geom_surfacepoint, geom_centroid, version
        FROM {schema}.{table_name}
        WHERE version = '{version}'
        """
        return pd.read_sql(text(query), self.engine)

    def get_mv_griddistrict(
        self,
        version: str = 'v0.4.5',
        schema: Literal['model_draft', 'data'] = 'data',
        table_name: str = 'ego_dp_mv_griddistrict'
    ) -> pd.DataFrame:
        """
        Load MV grid district data (replaces ego_dp_mv_griddistrict_*_mview).

        This function replicates the logic from:
        - mviews.ego_dp_mv_griddistrict_v0_4_3_mview
        - mviews.ego_dp_mv_griddistrict_v0_4_5_mview

        Parameters
        ----------
        version : str, default 'v0.4.5'
            Version string
        schema : str, default 'data'
            Database schema name
        table_name : str, default 'ego_dp_mv_griddistrict'
            Base table name

        Returns
        -------
        pd.DataFrame
            MV grid district data
        """
        query = f"""
        SELECT
            subst_id, subst_name, ags_0, geom, version
        FROM {schema}.{table_name}
        WHERE version = '{version}'
        """
        return pd.read_sql(text(query), self.engine)

    def load_mview_replacement(
        self,
        mview_name: str,
        scenario: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generic mview replacement loader based on mview name.

        This is a convenience function that automatically routes to the appropriate
        specific function based on the mview name pattern.

        Parameters
        ----------
        mview_name : str
            Name of the mview to replace (e.g., 'ego_dp_conv_powerplant_nep2035_mview')
        scenario : str, optional
            Scenario name (extracted from mview_name if not provided)
        version : str, optional
            Version string
        **kwargs
            Additional arguments passed to specific functions

        Returns
        -------
        pd.DataFrame
            Data from the appropriate base table

        Raises
        ------
        ValueError
            If the mview_name pattern is not recognized

        Examples
        --------
        >>> replacer = MviewsReplacement(engine)
        >>> data = replacer.load_mview_replacement('ego_dp_conv_powerplant_nep2035_mview')
        >>> data = replacer.load_mview_replacement('ego_supply_res_powerplant_ego100_mview')
        """
        # Remove 'mviews.' prefix if present
        mview_name = mview_name.replace('mviews.', '')

        # Determine schema from mview name
        if mview_name.startswith('ego_supply_'):
            schema = kwargs.get('schema', 'data')
            base_prefix = 'ego_dp_supply_'
        else:
            schema = kwargs.get('schema', 'data')
            base_prefix = 'ego_dp_'

        # Parse conventional power plants
        if 'conv_powerplant' in mview_name:
            # Extract scenario from name if not provided
            if scenario is None:
                if 'ego100' in mview_name or 'ego_100' in mview_name:
                    scenario = 'eGo 100'
                elif 'nep2035' in mview_name:
                    scenario = 'NEP 2035'
                elif 'sq' in mview_name or 'status' in mview_name:
                    scenario = 'Status Quo'
                else:
                    raise ValueError(f"Cannot determine scenario from mview name: {mview_name}")

            # Determine table name
            if mview_name.startswith('ego_supply_'):
                table_name = 'ego_dp_supply_conv_powerplant'
            else:
                table_name = 'ego_dp_conv_powerplant'

            return self.get_conv_powerplant(
                scenario=scenario,
                version=version,
                schema=schema,
                table_name=table_name,
                **kwargs
            )

        # Parse renewable power plants
        elif 'res_powerplant' in mview_name:
            if scenario is None:
                if 'ego100' in mview_name or 'ego_100' in mview_name:
                    scenario = 'eGo 100'
                elif 'nep2035' in mview_name:
                    scenario = 'NEP 2035'
                elif 'sq' in mview_name or 'status' in mview_name:
                    scenario = 'Status Quo'
                else:
                    raise ValueError(f"Cannot determine scenario from mview name: {mview_name}")

            # Determine table name
            if mview_name.startswith('ego_supply_'):
                table_name = 'ego_dp_supply_res_powerplant'
            else:
                table_name = 'ego_dp_res_powerplant'

            return self.get_res_powerplant(
                scenario=scenario,
                version=version,
                schema=schema,
                table_name=table_name,
                **kwargs
            )

        # Parse load areas
        elif 'loadarea' in mview_name:
            # Extract version from name if not provided
            if version is None:
                if 'v0_4_5' in mview_name or 'v0.4.5' in mview_name:
                    version = 'v0.4.5'
                elif 'v0_4_3' in mview_name or 'v0.4.3' in mview_name:
                    version = 'v0.4.3'
                else:
                    version = 'v0.4.5'  # default

            return self.get_loadarea(
                version=version,
                schema=schema,
                **kwargs
            )

        # Parse MV grid districts
        elif 'mv_griddistrict' in mview_name:
            if version is None:
                if 'v0_4_5' in mview_name or 'v0.4.5' in mview_name:
                    version = 'v0.4.5'
                elif 'v0_4_3' in mview_name or 'v0.4.3' in mview_name:
                    version = 'v0.4.3'
                else:
                    version = 'v0.4.5'  # default

            return self.get_mv_griddistrict(
                version=version,
                schema=schema,
                **kwargs
            )

        else:
            raise ValueError(f"Unsupported mview name pattern: {mview_name}")


# Convenience function for backwards compatibility
def get_mview_data(engine, mview_name: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to load mview replacement data.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        Database engine
    mview_name : str
        Name of the mview to replace
    **kwargs
        Additional arguments passed to load_mview_replacement

    Returns
    -------
    pd.DataFrame
        Data from the appropriate base table

    Examples
    --------
    >>> from sqlalchemy import create_engine
    >>> engine = create_engine('postgresql://...')
    >>> data = get_mview_data(engine, 'ego_dp_conv_powerplant_nep2035_mview')
    """
    replacer = MviewsReplacement(engine)
    return replacer.load_mview_replacement(mview_name, **kwargs)
