import pandas as pd


class OverlyingGrid:
    """
    Data container for requirements from the overlying grid.

    The requirements from the overlying grid are used as constraints for flexibilities.

    Attributes
    -----------
    renewables_curtailment : :pandas:`pandas.Series<Series>`
        Curtailment of fluctuating renewables per time step in MW.
    storage_units_active_power : :pandas:`pandas.Series<Series>`
        Aggregated dispatch of storage units per time step in MW.
    dsm_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand side management utilisation per time step in MW.
    electromobility_active_power : :pandas:`pandas.Series<Series>`
        Aggregated charging demand at flexible charging sites per time step in MW.
    heat_pump_decentral_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand of flexible decentral heat pumps per time step in MW.
    heat_pump_central_active_power : :pandas:`pandas.Series<Series>`
        Aggregated demand of flexible central heat pumps per time step in MW.
    geothermal_energy_feedin_district_heating : :pandas:`pandas.DataFrame<DataFrame>`
        Geothermal feed-in into district heating per district heating area (in columns)
        and time step (in index) in MW.
    solarthermal_energy_feedin_district_heating : :pandas:`pandas.DataFrame<DataFrame>`
        Solarthermal feed-in into district heating per district heating area (in
        columns) and time step (in index) in MW.

    """

    def __init__(self, **kwargs):
        self.renewables_curtailment = kwargs.get(
            "renewables_curtailment", pd.Series(dtype="float64")
        )

        self.storage_units_active_power = kwargs.get(
            "storage_units_active_power", pd.Series(dtype="float64")
        )
        self.dsm_active_power = kwargs.get(
            "dsm_active_power", pd.Series(dtype="float64")
        )
        self.electromobility_active_power = kwargs.get(
            "electromobility_active_power", pd.Series(dtype="float64")
        )
        self.heat_pump_decentral_active_power = kwargs.get(
            "heat_pump_decentral_active_power", pd.Series(dtype="float64")
        )
        self.heat_pump_central_active_power = kwargs.get(
            "heat_pump_central_active_power", pd.Series(dtype="float64")
        )

        self.geothermal_energy_feedin_district_heating = kwargs.get(
            "geothermal_energy_feedin_district_heating", pd.DataFrame(dtype="float64")
        )
        self.solarthermal_energy_feedin_district_heating = kwargs.get(
            "solarthermal_energy_feedin_district_heating", pd.DataFrame(dtype="float64")
        )

    @property
    def _attributes(self):
        return [
            "renewables_curtailment",
            "storage_units_active_power",
            "dsm_active_power",
            "electromobility_active_power",
            "heat_pump_decentral_active_power",
            "heat_pump_central_active_power",
            "geothermal_energy_feedin_district_heating",
            "solarthermal_energy_feedin_district_heating",
        ]
