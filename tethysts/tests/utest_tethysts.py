from tethysts import Tethys
import yaml
import pandas as pd
import os

pd.options.display.max_columns = 10

##############################################
### Parameters

base_dir = os.path.split(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0])[0]

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)

remotes_list = param['remotes']

######################################
### Testing

# remote = {'bucket': 'fire-emergency-nz', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = remotes_list[0]
# remote['connection_config'] = 'https://b2.tethys-ts.xyz'
#
dataset_id = '74c5bcd07846abae0e28ddd2'
# dataset_id = '10456b32c1eb6f20339d16b4'
# station_id = 'f79f0ddc99428b73c2293513'
station_id = '6b75a7fb1453ef94148bda19'
# station_ids = [station_id, 'f74d29232b5d5c094effe9e2']
# dataset_id='74c5bcd07846abae0e28ddd2'
# station_id='fabdf416a8644a713e221fd6'
# dataset_id='ad3156ce8245f725a5a0cda8'
# station_id='6b5dc8c9ec394b2c4167f6b8'
#
#
# self = Tethys([remote])
self = Tethys(remotes_list)
#
stn_list1 = self.get_stations(dataset_id)
# run_dates = self.get_run_dates(dataset_id, station_id)
data1 = self.get_results(dataset_id, station_id, remove_height=True, output='Dataset')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, output='DataArray')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, remove_height=True, output='DataArray')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, output='Dict')
# data1 = self.get_results(dataset_id, station_id, output='Dict')
# data1 = self.get_results(dataset_id, station_id, from_date='2012-01-02 00:00', output='Dataset')

# data2 = self.get_bulk_results(dataset_id, station_ids, output='DataArray')

# dataset_id = 'f4cfb5a362707785dd39ff85'
# station_id = 'ff4213c61878e098e07df513'















