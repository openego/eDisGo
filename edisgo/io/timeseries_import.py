import pandas as pd
from egoio.db_tables import model_draft, supply

from edisgo.tools import session_scope


def import_feedin_timeseries(config_data, weather_cell_ids):
    """
    Import RES feed-in time series data and process

    Parameters
    ----------
    config_data : dict
        Dictionary containing config data from config files.
    weather_cell_ids : :obj:`list`
        List of weather cell id's (integers) to obtain feed-in data for.

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Feedin time series

    """

    def _retrieve_timeseries_from_oedb(session):
        """Retrieve time series from oedb

        """
        # ToDo: add option to retrieve subset of time series
        # ToDo: find the reference power class for mvgrid/w_id and insert instead of 4
        feedin_sqla = session.query(
            orm_feedin.w_id,
            orm_feedin.source,
            orm_feedin.feedin). \
            filter(orm_feedin.w_id.in_(weather_cell_ids)). \
            filter(orm_feedin.power_class.in_([0, 4])). \
            filter(orm_feedin_version)

        feedin = pd.read_sql_query(feedin_sqla.statement,
                                   session.bind,
                                   index_col=['source', 'w_id'])
        return feedin

    if config_data['data_source']['oedb_data_source'] == 'model_draft':
        orm_feedin_name = config_data['model_draft']['res_feedin_data']
        orm_feedin = model_draft.__getattribute__(orm_feedin_name)
        orm_feedin_version = 1 == 1
    else:
        orm_feedin_name = config_data['versioned']['res_feedin_data']
        orm_feedin = supply.__getattribute__(orm_feedin_name)
        orm_feedin_version = orm_feedin.version == config_data['versioned'][
            'version']

    with session_scope() as session:
        feedin = _retrieve_timeseries_from_oedb(session)

    feedin.sort_index(axis=0, inplace=True)

    timeindex = pd.date_range('1/1/2011', periods=8760, freq='H')

    recasted_feedin_dict = {}
    for type_w_id in feedin.index:
        recasted_feedin_dict[type_w_id] = feedin.loc[
                                          type_w_id, :].values[0]

    feedin = pd.DataFrame(recasted_feedin_dict, index=timeindex)

    # rename 'wind_onshore' and 'wind_offshore' to 'wind'
    new_level = [_ if _ not in ['wind_onshore']
                 else 'wind' for _ in feedin.columns.levels[0]]
    feedin.columns.set_levels(new_level, level=0, inplace=True)

    feedin.columns.rename('type', level=0, inplace=True)
    feedin.columns.rename('weather_cell_id', level=1, inplace=True)

    return feedin
