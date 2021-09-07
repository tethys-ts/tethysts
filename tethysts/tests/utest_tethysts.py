from tethysts import Tethys
import yaml
import pandas as pd
import os
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree
import pickle
from shapely import wkb, wkt
import orjson
import tethys_utils as tu

pd.options.display.max_columns = 10

##############################################
### Parameters

base_dir = os.path.split(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0])[0]

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)

remotes_list = param['remotes']

######################################
### Testing

remote = {'bucket': 'fire-emergency-nz', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'es-hilltop', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'niwa-cliflo', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'ecan-env-monitoring', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'nz-forecasts', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'met-solutions', 'connection_config': 'https://b2.tethys-ts.xyz'}
# remote = {'bucket': 'nasa-data', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'tethysts', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'nz-open-modelling-consortium', 'connection_config': 'https://b2.nzrivers.xyz'}

remote = remotes_list[-1]
# remote['connection_config'] = 'https://b2.tethys-ts.xyz'
#
# dataset_id = '4edc94c19bf074027bc7c099'
# dataset_id = 'dddb02cd5cb7ae191311ab19'
# station_id = 'fedeb59e6c7f47597a7d47c7'
# station_id = 'fe9a63fae6f7fe58474bb3c0'
# station_id2 = 'fe9a63fae6f7fe58474bb3c0'
# station_id = '6b75a7fb1453ef94148bda19'
# station_ids = [station_id, '5d06c5a8065a26b51c19b241']
dataset_id='361ce2acd56b13da82390a69'
station_id='00128a218015a069cb94d360'

dataset_id = '22a389416b05243e3957a113'

# dataset_id='320d6836250169a5f7b78163'
# station_id='7df0d7fe8c6fcd06c50d73a6'
station_id = 'de7cfd749e24a6b78c2281fb'
station_ids = [station_id, 'de7cfd749e24a6b78c2281fb']
dataset_id = '4690821a39b3fb65c197a540'
station_id = '78b7a3d3f61cf33e681490c1'
station_ids = [station_id, '9f3ee54af239b845241f8f13']
dataset_id = '38138ea1c3350031d1b217f6'
station_id = 'b318207aa246e7bbbd74cb19'

dataset_id = '4ae05d099af292fec48792ec'
station_id = 'dfb66ed0f4835161a7001d45'

dataset_id = 'fb77f37b16edae3534e73ddd'
station_id = 'fd60b5bafe19d6b243dda43d'

dataset_id = 'e37f1451fcf8f9e64b66be8d'

dataset_id = '9c7e107f99180e45eafdf5af'
station_id = 'c6df8b47b2efce3daedef48e'
station_id = 'a3f2d35fd6df8247cea32d03'

dataset_id = '9bf36a9e6b6a2a111bf6634b'
station_id = '4db28a9db0cb036507490887'

dataset_id = '9845cd0049891916f2a59c80'
station_id = '02d4943e784fcb6acd819b72'

dataset_id = '668373c15a01955128c95bbd'
station_id = '673d6d9fca3ccf38fa009ad1'
station_ids = [station_id, '67351e51dc55f730471248fc']

dataset_id = 'dddb02cd5cb7ae191311ab19'
station_id = 'fedeb59e6c7f47597a7d47c7'
#
#
self = Tethys([remote])
self = Tethys(remotes_list)
#
stn_list1 = self.get_stations(dataset_id)
stn_list1 = self.get_stations(dataset_id, results_object_keys=True)
# run_dates = self.get_run_dates(dataset_id, station_id)
data1 = self.get_results(dataset_id, station_id, output='Dataset')
data1 = self.get_results(dataset_id, station_id, output='Dataset', cache='memory')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, remove_height=True, output='DataArray')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, remove_height=True, output='DataArray')
# data1 = self.get_results(dataset_id, station_id, modified_date=True, quality_code=True, output='Dict')
# data1 = self.get_results(dataset_id, station_id, output='Dict')
data1 = self.get_results(dataset_id, station_id, run_date='2021-08-29T00:00:00', output='Dataset')
data1 = self.get_results(dataset_id, station_id, run_date='2021-03-12T18:00:16', output='Dataset')

data1 = self.get_results(dataset_id, station_id, squeeze_dims=True, output='DataArray', cache='memory')
data2 = self.get_results(dataset_id, station_id2, output='DataArray')

run_dates1 = self.get_run_dates(dataset_id, station_id)

station_ids = [s['station_id'] for s in stn_list1]
data2 = self.get_bulk_results(dataset_id, station_ids, output='Dataset', threads=10)

# dataset_id = 'f4cfb5a362707785dd39ff85'
# station_id = 'ff4213c61878e098e07df513'

query_geometry = {'type': 'Point', 'coordinates': [171.043868, -43.106372]}
geom_query = shape(query_geometry).buffer(0.1)

stn_list2 = self.get_stations(dataset_id, query_geometry)
stn_list2 = self.get_stations(dataset_id, lat=-43.1, lon=171.1)
data2 = self.get_results(dataset_id, geometry=query_geometry, squeeze_dims=True, output='Dataset')

stn = [s for s in stn_list1 if s['station_id'] == station_id]

stn = [s for s in stn_list1 if 'Waiau River' in s['ref']]


gwl_ds1 = [d for d in self.datasets if d['parameter'] == 'groundwater_depth']
era5_ds1 = [d for d in self.datasets if d['owner'] == 'ECMWF']


stns = self.get_stations(dataset_id)

type1 = stns[0]['geometry']['type']

if type1 == 'Polygon':
    geo = [s['geometry']['coordinates'][0][0][-1] for s in stns]
    geo1 = np.array(geo).round(5)
    geo1.sort()
    np.diff(geo1)












