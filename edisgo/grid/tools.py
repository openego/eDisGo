import logging
logger = logging.getLogger('edisgo')


def select_cable(network, level, apparent_power):
    """Selects an appropriate cable type and quantity using given apparent power.

    Considers load factor.

    Parameters
    ----------
    network : :class:`~.grid.network.Network`
        The eDisGo container object
    level : :obj:`str`
        Grid level ('mv' or 'lv')
    apparent_power : :obj:`float`
        Apparent power the cable must carry in kVA

    Returns
    -------
    :pandas:`pandas.Series<series>`
        Cable type
    :obj:`ìnt`
        Cable count
    """

    cable_count = 1

    if level == 'mv':
        load_factor = network.config['grid_expansion']['load_factor_mv_line']

        available_cables = network.equipment_data['MV_cables'][
            network.equipment_data['MV_cables']['U_n'] == network.mv_grid.voltage_nom]

        suitable_cables = available_cables[
            available_cables['I_max_th'] *
            network.mv_grid.voltage_nom *
            load_factor > apparent_power]

        # increase cable count until appropriate cable type is found
        while suitable_cables.empty:
            cable_count += 1
            suitable_cables = available_cables[
                available_cables['I_max_th'] *
                network.mv_grid.voltage_nom *
                cable_count *
                load_factor > apparent_power]

        cable_type = suitable_cables.ix[suitable_cables['I_max_th'].idxmin()]

    elif level == 'lv':
        load_factor = network.config['grid_expansion']['load_factor_lv_line']

        suitable_cables = network.equipment_data['LV_cables'][
            network.equipment_data['LV_cables']['I_max_th'] *
            network.equipment_data['LV_cables']['U_n'] *
            load_factor > apparent_power]

        # increase cable count until appropriate cable type is found
        while suitable_cables.empty:
            cable_count += 1
            suitable_cables = network.equipment_data['LV_cables'][
                network.equipment_data['LV_cables']['I_max_th'] *
                network.equipment_data['LV_cables']['U_n'] *
                cable_count *
                load_factor > apparent_power]

        cable_type = suitable_cables.ix[suitable_cables['I_max_th'].idxmin()]

    else:
        raise ValueError('Please supply a level (either \'mv\' or \'lv\').')

    return cable_type, cable_count
