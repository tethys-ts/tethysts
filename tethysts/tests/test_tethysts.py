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

remote1 = {'bucket': 'ecan-env-monitoring', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 2}

remote2 = {'bucket': 'nz-open-modelling-consortium', 'public_url': 'https://b2.nzrivers.xyz/file/', 'version': 3}

remote3 = {'bucket': 'nz-open-modelling-consortium', 'public_url': 'https://b2.nzrivers.xyz/file/', 'version': 4}

dataset_id = 'f5b878a968d2c5bc404c07e5'
station_ids = '9b46e9569369e969f6946ba3'

# dataset_id = 'c3a09c8a5da175897916e8e8'
# station_ids = '4db28a9db0cb036507490887'

remotes = [
    {'remote': remote1,
    'dataset_id': 'c3a09c8a5da175897916e8e8',
    'station_ids': '4db28a9db0cb036507490887',
    'assert':
        {'datasets': 1,
          'stations': 1,
          'versions': 0,
          'results': 1,
          }
    },
    {'remote': remote2,
      'dataset_id': 'f27574a7b38eab5b0bc2b3d7',
      'station_ids': '9c90243e84b8c5b17f0726c4',
      'assert':
          {'datasets': 1,
          'stations': 1,
          'versions': 0,
          'results': 1,
          }
      },
    {'remote': remote3,
     'dataset_id': dataset_id,
     'station_ids': station_ids,
     'assert':
         {'datasets': 1,
          'stations': 1,
          'versions': 0,
          'results': 1,
          }
     },
     ]

outputs = ['Dataset', 'DataArray', 'Dict']

geometry1 = {'type': 'Point', 'coordinates': [172, -42.8]}
# geometry2 = shape(geometry1).buffer(0.5)
lat = -42.8
lon = 172.0
distance = 0.1

######################################
### Testing


@pytest.mark.parametrize('remote', remotes)
def test_tethys(remote):
    """

    """
    t1 = Tethys([remote['remote']])

    ## Datasets
    datasets = t1.datasets
    assert len(datasets) > remote['assert']['datasets']

    ## Stations
    stn_list1 = t1.get_stations(remote['dataset_id'])
    assert len(stn_list1) > remote['assert']['stations']

    ## Versions
    rv1 = t1.get_versions(remote['dataset_id'])
    assert len(rv1) > remote['assert']['versions']

    ## Results
    data1 = t1.get_results(remote['dataset_id'], remote['station_ids'], output='Dataset')
    assert len(data1) > remote['assert']['results']


## initialise for the rest of the tests
t1 = Tethys([remote3])


@pytest.mark.parametrize('output', outputs)
def test_get_results(output):
    data1 = t1.get_results(dataset_id, station_ids, squeeze_dims=True, output=output)

    if output == 'Dataset':
        assert len(data1.time) > 90
    elif output == 'DataArray':
        assert len(data1.time) > 90
    elif output == 'Dict':
        assert len(data1['coords']['time']['data']) > 90
    else:
        raise ValueError('Forgot to put in new assertion')


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

    assert len(s1) > 1


def test_get_nearest_results2():
    s2 = t1.get_results(dataset_id, lat=lat, lon=lon)

    assert len(s2) > 1


# def test_get_intersection_results1():
#     s3 = t1.get_results(dataset_id, lat=lat, lon=lon, distance=distance)
#
#     assert len(s3) > 1
