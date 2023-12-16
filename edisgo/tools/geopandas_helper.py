from __future__ import annotations

import os

from typing import TYPE_CHECKING

if "READTHEDOCS" not in os.environ:
    import geopandas as gpd

    from shapely.geometry import LineString

if TYPE_CHECKING:
    from edisgo.network.grids import Grid

COMPONENTS: list[str] = [
    "generators_df",
    "loads_df",
    "storage_units_df",
    "transformers_df",
]


class GeoPandasGridContainer:
    """
    Grids geo data for all components with information about their geolocation.

    Parameters
    ----------
    crs : str
        Coordinate Reference System of the geometry objects.
    id : str or int
        Grid identifier
    grid : :class:`~.network.grids.Grid`
        Matching grid object
    buses_gdf : :geopandas:`GeoDataFrame`
        GeoDataframe with all buses in the Grid. See
        :attr:`~.network.topology.Topology.buses_df` for more information.
    generators_gdf : :geopandas:`GeoDataFrame`
        GeoDataframe with all generators in the Grid. See
        :attr:`~.network.topology.Topology.generators_df` for more information.
    loads_gdf : :geopandas:`GeoDataFrame`
        GeoDataframe with all loads in the Grid. See
        :attr:`~.network.topology.Topology.loads_df` for more information.
    storage_units_gdf : :geopandas:`GeoDataFrame`
        GeoDataframe with all storage units in the Grid. See
        :attr:`~.network.topology.Topology.storage_units_df` for more information.
    transformers_gdf : :geopandas:`GeoDataFrame`
        GeoDataframe with all transformers in the Grid. See
        :attr:`~.network.topology.Topology.transformers_df` for more information.
    lines_gdf : :geopandas:`GeoDataFrame`
        GeoDataframe with all lines in the Grid. See
        :attr:`~.network.topology.Topology.loads_df` for more information.
    """

    def __init__(
        self,
        crs: str,
        grid_id: str | int,
        grid: Grid,
        buses_gdf: gpd.GeoDataFrame,
        generators_gdf: gpd.GeoDataFrame,
        loads_gdf: gpd.GeoDataFrame,
        storage_units_gdf: gpd.GeoDataFrame,
        transformers_gdf: gpd.GeoDataFrame,
        lines_gdf: gpd.GeoDataFrame,
    ):
        self.crs = crs
        self.grid_id = grid_id
        self.grid = grid
        self.buses_gdf = buses_gdf
        self.generators_gdf = generators_gdf
        self.loads_gdf = loads_gdf
        self.storage_units_gdf = storage_units_gdf
        self.transformers_gdf = transformers_gdf
        self.lines_gdf = lines_gdf

        @property
        def crs(self):
            """The crs property."""
            return self._crs

        @crs.setter
        def crs(self, crs_str):
            self._crs = crs_str

        @property
        def grid_id(self):
            """The grid_id property."""
            return self._grid_id

        @grid_id.setter
        def grid_id(self, grid_id_val):
            self._grid_id = grid_id_val

        @property
        def grid(self):
            """The grid property."""
            return self.grid

        @grid.setter
        def grid(self, grid_obj):
            self.grid = grid_obj

        @property
        def buses_gdf(self):
            """The buses_gdf property."""
            return self._buses_gdf

        @buses_gdf.setter
        def buses_gdf(self, gdf):
            self._buses_gdf = gdf

        @property
        def generators_gdf(self):
            """The generators_gdf property."""
            return self._generators_gdf

        @generators_gdf.setter
        def generators_gdf(self, gdf):
            self._generators_gdf = gdf

        @property
        def loads_gdf(self):
            """The loads_gdf property."""
            return self._loads_gdf

        @loads_gdf.setter
        def loads_gdf(self, gdf):
            self._loads_gdf = gdf

        @property
        def storage_units_gdf(self):
            """The storage_units_gdf property."""
            return self._storage_units_gdf

        @storage_units_gdf.setter
        def storage_units_gdf(self, gdf):
            self._storage_units_gdf = gdf

        @property
        def transformers_gdf(self):
            """The transformers_gdf property."""
            return self._transformers_gdf

        @transformers_gdf.setter
        def transformers_gdf(self, gdf):
            self._transformers_gdf = gdf

        @property
        def lines_gdf(self):
            """The lines_gdf property."""
            return self._lines_gdf

        @lines_gdf.setter
        def lines_gdf(self, gdf):
            self._lines_gdf = gdf

        def plot(self):
            """
            TODO: Implement plotting functions as needed
            """
            raise NotImplementedError


def to_geopandas(grid_obj: Grid):
    """
    Translates all DataFrames with geolocations within a Grid class to GeoDataFrames.

    Parameters
    ----------
    grid_obj : :class:`~.network.grids.Grid`
        Grid object to transform.

    Returns
    -------
    :class:`.GeoPandasGridContainer`
        Data container with the grids geo data for all components with information about
        their geolocation.

    """
    # get srid id
    srid = grid_obj._edisgo_obj.topology.grid_district["srid"]

    # convert buses_df
    buses_df = grid_obj.buses_df
    buses_df = buses_df.assign(
        geometry=gpd.points_from_xy(buses_df.x, buses_df.y, crs=f"EPSG:{srid}")
    ).drop(columns=["x", "y"])

    buses_gdf = gpd.GeoDataFrame(buses_df, crs=f"EPSG:{srid}")

    # convert component DataFrames
    components_dict = {}

    for component in COMPONENTS:
        left_on = "bus1" if component == "transformers_df" else "bus"

        attr = getattr(grid_obj, component)

        components_dict[component.replace("_df", "_gdf")] = gpd.GeoDataFrame(
            attr.merge(
                buses_gdf[["geometry", "v_nom"]], left_on=left_on, right_index=True
            ),
            crs=f"EPSG:{srid}",
        )
        if components_dict[component.replace("_df", "_gdf")].empty:
            components_dict[component.replace("_df", "_gdf")].index = components_dict[
                component.replace("_df", "_gdf")
            ].index.astype(object)

    # convert lines_df
    lines_df = grid_obj.lines_df

    geom_0 = lines_df.merge(
        buses_gdf[["geometry"]], left_on="bus0", right_index=True
    ).geometry
    geom_1 = lines_df.merge(
        buses_gdf[["geometry"]], left_on="bus1", right_index=True
    ).geometry

    geometry = [
        LineString([point_0, point_1]) for point_0, point_1 in list(zip(geom_0, geom_1))
    ]

    lines_gdf = gpd.GeoDataFrame(lines_df.assign(geometry=geometry), crs=f"EPSG:{srid}")

    return GeoPandasGridContainer(
        crs=f"EPSG:{srid}",
        grid_id=grid_obj.id,
        grid=grid_obj,
        buses_gdf=buses_gdf,
        lines_gdf=lines_gdf,
        **components_dict,
    )
