"""
Created on 2021-04-27.

@author: Mike K
"""
from tethysts import Tethys
import pandas as pd
import os
import pytest

pd.options.display.max_columns = 10

##############################################
### Parameters

remote = {'bucket': 'fire-emergency-nz', 'connection_config': 'https://b2.tethys-ts.xyz'}

dataset_id = 'dddb02cd5cb7ae191311ab19'
station_id = 'fedeb59e6c7f47597a7d47c7'
station_ids = [station_id, 'fe9a63fae6f7fe58474bb3c0']

outputs = ['Dataset', 'DataArray', 'Dict']

######################################
### Testing


def test_get_datasets():
    t1 = Tethys([remote])
    datasets = t1.datasets
    assert len(datasets) > 5


## initialise for the rest of the tests
t1 = Tethys([remote])


def test_get_stations():
    stn_list1 = t1.get_stations(dataset_id)

    assert len(stn_list1) > 200


def test_get_run_dates():
    run_dates = t1.get_run_dates(dataset_id, station_id)

    assert len(run_dates) > 1


@pytest.mark.parametrize('output', outputs)
def test_get_results(output):
    data1 = t1.get_results(dataset_id, station_id, remove_height=True, output=output)

    assert len(data1) > 3


def test_get_bulk_results():
    data2 = t1.get_bulk_results(dataset_id, station_ids, remove_height=True, output='Dataset')

    assert len(data2) > 6












