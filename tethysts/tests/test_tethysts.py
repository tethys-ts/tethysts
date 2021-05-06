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

remote = {'bucket': 'ecan-env-monitoring', 'connection_config': 'https://b2.tethys-ts.xyz'}

dataset_id = 'b5d84aa773de2a747079c127'
station_id = '4db28a9db0cb036507490887'
station_ids = [station_id, '474f75b4de127caca088620a']

outputs = ['Dataset', 'DataArray', 'Dict']

geometry1 = {'type': 'Point', 'coordinates': [172, -42.8]}
# geometry2 = shape(geometry1).buffer(0.5)
lat = -42.8
lon = 172.0
distance = 0.2

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

    assert len(stn_list1) > 2


## Get the stations loaded in the object
stn_list1 = t1.get_stations(dataset_id)


def test_get_run_dates():
    run_dates = t1.get_run_dates(dataset_id, station_id)

    assert len(run_dates) > 1


@pytest.mark.parametrize('output', outputs)
def test_get_results(output):
    data1 = t1.get_results(dataset_id, station_id, squeeze_dims=True, output=output)

    if output == 'Dataset':
        assert len(data1.time) > 90
    elif output == 'DataArray':
        assert len(data1.time) > 90
    elif output == 'Dict':
        assert len(data1['coords']['time']['data']) > 90
    else:
        raise ValueError('Forgot to put in new assertion')


def test_get_bulk_results():
    data2 = t1.get_bulk_results(dataset_id, station_ids, squeeze_dims=True, output='Dataset')

    assert len(data2) > 6


def test_get_nearest_station1():
    s1 = t1.get_stations(dataset_id, geometry1)

    assert len(s1) == 1


def test_get_nearest_station2():
    s2 = t1.get_stations(dataset_id, lat=lat, lon=lon)

    assert len(s2) == 1


def test_get_intersection_stations1():
    s3 = t1.get_stations(dataset_id, lat=lat, lon=lon, distance=distance)

    assert len(s3) >= 2


def test_get_nearest_results1():
    s1 = t1.get_results(dataset_id, geometry=geometry1)

    assert len(s1) == 7


def test_get_nearest_results2():
    s2 = t1.get_results(dataset_id, lat=lat, lon=lon)

    assert len(s2) == 7


# def test_get_intersection_stations1():
#     s3 = t1.get_results(dataset_id, lat=lat, lon=lon, distance=distance)

#     assert len(s3) >= 3



































