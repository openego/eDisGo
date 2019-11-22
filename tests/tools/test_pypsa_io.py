import pytest
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np

from edisgo.tools.pypsa_io import append_lv_components, \
    get_timeseries_with_aggregated_elements


class TestPypsaIO:

    def test_append_lv_components(self):
        lv_components = {'Load': pd.DataFrame(),
                         'Generator': pd.DataFrame(),
                         'StorageUnit': pd.DataFrame()}
        comps = pd.DataFrame({'bus':[]})
        # check if returns when comps is empty
        append_lv_components('Unkown', comps, lv_components,
                             'TestGrid')
        # check exceptions for wrong input parameters
        comps = pd.DataFrame({'bus': ['bus1']}, index=['dummy'])
        msg = 'Component type not defined.'
        with pytest.raises(ValueError, match=msg):
            append_lv_components('Unkown', comps, lv_components,
                                 'TestGrid')
        msg = 'Aggregation type for loads invalid.'
        with pytest.raises(ValueError, match=msg):
            append_lv_components('Load', comps, lv_components,
                                 'TestGrid', aggregate_loads='unknown')
        msg = 'Aggregation type for generators invalid.'
        with pytest.raises(ValueError, match=msg):
            append_lv_components('Generator', comps, lv_components, 'TestGrid',
                                 aggregate_generators='unknown')
        msg = 'Aggregation type for storages invalid.'
        with pytest.raises(ValueError, match=msg):
            append_lv_components('StorageUnit', comps, lv_components,
                                 'TestGrid', aggregate_storages='unknown')
        # check appending aggregated elements to lv_components in different
        # modes
        # CHECK GENERATORS
        gens = pd.DataFrame({'bus': ['LVStation'] * 6, 'control': ['PQ'] * 6,
                             'p_nom': [0.05, 0.23, 0.04, 0.2, 0.1, 0.4],
                             'type': ['solar', 'wind', 'solar', 'solar', 'gas',
                                      'wind']}, index=['Solar_1', 'Wind_1',
                                                       'Solar_2', 'Solar_3',
                                                       'Gas_1', 'Wind_2'])
        # check not aggregated generators
        aggr_dict = append_lv_components('Generator', gens, lv_components,
                                         'TestGrid', aggregate_generators=None)
        assert len(aggr_dict) == 0
        assert len(lv_components['Generator']) == 6
        assert_frame_equal(gens.loc[:, ['bus', 'control', 'p_nom']],
                           lv_components['Generator'])
        # check aggregation of generators by type
        lv_components['Generator'] = pd.DataFrame()
        aggr_dict = append_lv_components('Generator', gens, lv_components,
                                         'TestGrid',
                                         aggregate_generators='type')
        assert len(aggr_dict) == 3
        assert (aggr_dict['TestGrid_gas'] == ['Gas_1']).all()
        assert (aggr_dict['TestGrid_solar'] ==
                ['Solar_1', 'Solar_2', 'Solar_3']).all()
        assert (aggr_dict['TestGrid_wind'] ==
                ['Wind_1', 'Wind_2']).all()
        assert len(lv_components['Generator']) == 3
        assert (lv_components['Generator'].control == 'PQ').all()
        assert (lv_components['Generator'].bus == 'LVStation').all()
        assert (lv_components['Generator'].index.values ==
                ['TestGrid_gas', 'TestGrid_solar', 'TestGrid_wind']).all()
        assert np.isclose(lv_components['Generator'].p_nom,
                          [0.1, 0.29, 0.63]).all()
        # check if only one type is existing
        lv_components['Generator'] = pd.DataFrame()
        aggr_dict = append_lv_components('Generator',
                                         gens.loc[gens.type == 'solar'],
                                         lv_components, 'TestGrid',
                                         aggregate_generators='type')
        assert len(aggr_dict) == 1
        assert (aggr_dict['TestGrid_solar'] ==
                ['Solar_1', 'Solar_2', 'Solar_3']).all()
        assert len(lv_components['Generator']) == 1
        assert lv_components['Generator'].index.values == ['TestGrid_solar']
        assert np.isclose(lv_components['Generator'].p_nom, 0.29)
        # check aggregation of generators by fluctuating or dispatchable
        lv_components['Generator'] = pd.DataFrame()
        aggr_dict = append_lv_components('Generator', gens, lv_components,
                                         'TestGrid',
                                         aggregate_generators='curtailable')
        assert len(aggr_dict) == 2
        assert (aggr_dict['TestGrid_fluctuating'] ==
                ['Solar_1', 'Wind_1', 'Solar_2', 'Solar_3', 'Wind_2']).all()
        assert (aggr_dict['TestGrid_dispatchable'] == ['Gas_1']).all()
        assert len(lv_components['Generator']) == 2
        assert (lv_components['Generator'].control == 'PQ').all()
        assert (lv_components['Generator'].bus == 'LVStation').all()
        assert (lv_components['Generator'].index.values ==
                ['TestGrid_fluctuating', 'TestGrid_dispatchable']).all()
        assert np.isclose(lv_components['Generator'].p_nom,
                          [0.92, 0.1]).all()
        # check if only dispatchable gens are given
        lv_components['Generator'] = pd.DataFrame()
        aggr_dict = append_lv_components('Generator',
                                         gens.loc[gens.type == 'gas'],
                                         lv_components, 'TestGrid',
                                         aggregate_generators='curtailable')
        assert len(aggr_dict) == 1
        assert (aggr_dict['TestGrid_dispatchable'] == ['Gas_1']).all()
        assert len(lv_components['Generator']) == 1
        assert lv_components['Generator'].index.values == \
               ['TestGrid_dispatchable']
        assert np.isclose(lv_components['Generator'].p_nom, 0.1)
        # check if only fluctuating gens are given
        lv_components['Generator'] = pd.DataFrame()
        aggr_dict = \
            append_lv_components('Generator',
                                 gens.drop(gens.loc[gens.type == 'gas'].index),
                                 lv_components, 'TestGrid',
                                 aggregate_generators='curtailable')
        assert len(aggr_dict) == 1
        assert (aggr_dict['TestGrid_fluctuating'] ==
                ['Solar_1', 'Wind_1', 'Solar_2', 'Solar_3', 'Wind_2']).all()
        assert len(lv_components['Generator']) == 1
        assert lv_components['Generator'].index.values == \
               ['TestGrid_fluctuating']
        assert np.isclose(lv_components['Generator'].p_nom, 0.92)
        # check aggregation of all generators
        lv_components['Generator'] = pd.DataFrame()
        aggr_dict = append_lv_components('Generator', gens, lv_components,
                                         'TestGrid', aggregate_generators='all')
        assert len(aggr_dict) == 1
        assert (aggr_dict['TestGrid_generators']
                == ['Solar_1', 'Wind_1', 'Solar_2', 'Solar_3', 'Gas_1',
                    'Wind_2']).all()
        assert len(lv_components['Generator']) == 1
        assert (lv_components['Generator'].control == 'PQ').all()
        assert (lv_components['Generator'].bus == 'LVStation').all()
        assert (lv_components['Generator'].index.values ==
                ['TestGrid_generators']).all()
        assert np.isclose(lv_components['Generator'].p_nom, 1.02)
        lv_components['Generator'] = pd.DataFrame()
        # CHECK LOADS
        loads = pd.DataFrame({'bus': ['LVStation'] * 6,
                              'peak_load': [0.05, 0.23, 0.04, 0.2, 0.1, 0.4],
                              'sector': ['retail', 'agricultural', 'retail',
                                       'retail', 'industrial', 'agricultural']},
                             index=['Retail_1', 'Agricultural_1', 'Retail_2',
                                    'Retail_3', 'Industrial_1',
                                    'Agricultural_2'])
        # check not aggregated loads
        aggr_dict = append_lv_components('Load', loads, lv_components,
                                         'TestGrid', aggregate_loads=None)
        assert len(aggr_dict) == 0
        assert len(lv_components['Load']) == 6
        assert (loads.peak_load.values ==
                lv_components['Load'].p_set.values).all()
        assert (lv_components['Load'].bus == 'LVStation').all()
        assert (lv_components['Load'].index == loads.index).all()
        # check aggregate loads by sector
        lv_components['Load'] = pd.DataFrame()
        aggr_dict = append_lv_components('Load', loads, lv_components,
                                         'TestGrid', aggregate_loads='sectoral')
        assert len(aggr_dict) == 3
        assert (aggr_dict['TestGrid_agricultural'] ==
                ['Agricultural_1', 'Agricultural_2']).all()
        assert (aggr_dict['TestGrid_industrial'] ==
                ['Industrial_1']).all()
        assert (aggr_dict['TestGrid_retail'] ==
                ['Retail_1', 'Retail_2', 'Retail_3']).all()
        assert len(lv_components['Load']) == 3
        assert (lv_components['Load'].bus == 'LVStation').all()
        assert (lv_components['Load'].index.values ==
                ['TestGrid_agricultural', 'TestGrid_industrial',
                 'TestGrid_retail']).all()
        assert np.isclose(lv_components['Load'].p_set, [0.63, 0.1, 0.29]).all()
        # check if only one sector exists
        lv_components['Load'] = pd.DataFrame()
        aggr_dict = append_lv_components('Load',
                                         loads.loc[loads.sector == 'industrial'],
                                         lv_components, 'TestGrid',
                                         aggregate_loads='sectoral')
        assert len(aggr_dict) == 1
        assert (aggr_dict['TestGrid_industrial'] ==
                ['Industrial_1']).all()
        assert len(lv_components['Load']) == 1
        assert (lv_components['Load'].bus == 'LVStation').all()
        assert (lv_components['Load'].index.values ==
                ['TestGrid_industrial']).all()
        assert np.isclose(lv_components['Load'].p_set, 0.1).all()
        # check aggregation of all loads
        lv_components['Load'] = pd.DataFrame()
        aggr_dict = append_lv_components('Load', loads, lv_components,
                                         'TestGrid', aggregate_loads='all')
        assert len(aggr_dict) == 1
        assert (aggr_dict['TestGrid_loads'] ==
                ['Retail_1', 'Agricultural_1', 'Retail_2', 'Retail_3',
                 'Industrial_1', 'Agricultural_2']).all()
        assert len(lv_components['Load']) == 1
        assert (lv_components['Load'].bus == 'LVStation').all()
        assert (lv_components['Load'].index.values ==
                ['TestGrid_loads']).all()
        assert np.isclose(lv_components['Load'].p_set, 1.02).all()
        lv_components['Load'] = pd.DataFrame()
        # CHECK STORAGES
        storages = pd.DataFrame({'bus': ['LVStation'] * 2,
                                'control': ['PQ'] * 2},
                                index=['Storage_1', 'Storage_2'])
        # check appending without aggregation
        aggr_dict = append_lv_components('StorageUnit', storages,
                                         lv_components, 'TestGrid',
                                         aggregate_storages=None)
        assert len(aggr_dict) == 0
        assert len(lv_components['StorageUnit']) == 2
        assert (lv_components['StorageUnit'].bus == 'LVStation').all()
        assert (lv_components['StorageUnit'].control == 'PQ').all()
        assert (lv_components['StorageUnit'].index.values
                == ['Storage_1', 'Storage_2']).all()
        # check aggregration of all storages
        lv_components['StorageUnit'] = pd.DataFrame()
        aggr_dict = append_lv_components('StorageUnit', storages,
                                         lv_components, 'TestGrid',
                                         aggregate_storages='all')
        assert len(aggr_dict) == 1
        assert (aggr_dict['TestGrid_storages'] ==
                ['Storage_1', 'Storage_2']).all()
        assert len(lv_components['StorageUnit']) == 1
        assert (lv_components['StorageUnit'].bus == 'LVStation').all()
        assert (lv_components['StorageUnit'].control == 'PQ').all()
        assert lv_components['StorageUnit'].index.values == 'TestGrid_storages'

    def test_get_generators_timeseries_with_aggregated_elements(self):
        print()
        # Todo: implement
