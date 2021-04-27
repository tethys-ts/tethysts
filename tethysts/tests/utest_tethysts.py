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
dataset_id = '4edc94c19bf074027bc7c099'
dataset_id = 'dddb02cd5cb7ae191311ab19'
station_id = 'fedeb59e6c7f47597a7d47c7'
# station_id = 'fe9a63fae6f7fe58474bb3c0'
station_id = '6b75a7fb1453ef94148bda19'
# station_ids = [station_id, 'fe9a63fae6f7fe58474bb3c0']
# dataset_id='73ab8b02fc65686636eb6d0b'
# station_id='489de57745076fb4ff2bd91b'
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
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, remove_height=True, output='DataArray')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, remove_height=True, output='DataArray')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, output='Dict')
# data1 = self.get_results(dataset_id, station_id, output='Dict')
# data1 = self.get_results(dataset_id, station_id, from_date='2012-01-02 00:00', output='Dataset')

# data2 = self.get_bulk_results(dataset_id, station_ids, remove_height=True, output='DataArray')

# dataset_id = 'f4cfb5a362707785dd39ff85'
# station_id = 'ff4213c61878e098e07df513'

stn = [s for s in stn_list1 if s['ref'] == '1071103']












