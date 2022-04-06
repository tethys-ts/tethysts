from tethysts import Tethys
import yaml
import pandas as pd
import os
import shapely
import geopandas as gpd
from shapely.geometry import shape
from shapely.strtree import STRtree
import pickle
from shapely import wkb, wkt
import orjson
import tethys_utils as tu
import numpy as np

pd.options.display.max_columns = 10

##############################################
### Parameters

base_dir = os.path.split(os.path.split(os.path.realpath(os.path.dirname(__file__)))[0])[0]

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)

remotes_list = param['remotes']

######################################
### Testing

remote = {'bucket': 'fire-emergency-nz', 'connection_config': 'https://b2.tethys-ts.xyz', 'version': 2}
remote = {'bucket': 'es-hilltop', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'niwa-cliflo', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'ecan-env-monitoring', 'public_url': 'https://b2.tethys-ts.xyz/file', 'version': 2}
remote = {'bucket': 'point-forecasts', 'connection_config': 'https://b2.tethys-ts.xyz', 'version': 3}
remote = {'bucket': 'met-solutions', 'connection_config': 'https://b2.tethys-ts.xyz', 'version': 3}
# remote = {'bucket': 'nasa-data', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'tethysts', 'connection_config': 'https://b2.tethys-ts.xyz', 'version': 3}
remote = {'bucket': 'nz-open-modelling-consortium', 'connection_config': 'https://b2.nzrivers.xyz', 'version': 3}
remote = {'bucket': 'typhon', 'public_url': 'https://b2.tethys-ts.xyz', 'version': 4}

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
dataset_id = 'dddb02cd5cb7ae191311ab19'
station_ids = ['fedeb59e6c7f47597a7d47c7', 'fe9a63fae6f7fe58474bb3c0']

dataset_id='0b2bd62cc42f3096136f11e9'
station_id='e0c38cc6fd3eb51fb553d45a'

dataset_id = 'de3bff8e3c3a2ad9200d8684'
station_id = 'fedeb59e6c7f47597a7d47c7'

# dataset_id='320d6836250169a5f7b78163'
# station_id='7df0d7fe8c6fcd06c50d73a6'
station_id = 'de7cfd749e24a6b78c2281fb'
station_ids = [station_id, 'de7cfd749e24a6b78c2281fb']
dataset_id = '4690821a39b3fb65c197a540'
station_id = '78b7a3d3f61cf33e681490c1'
station_ids = [station_id, '9f3ee54af239b845241f8f13']
dataset_id = '38138ea1c3350031d1b217f6'
station_id = 'b318207aa246e7bbbd74cb19'

dataset_id = 'f56892eb59d12cfbc02acceb'
station_id = 'f4da7cc7f4c947b2e6b30344'

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

dataset_id = '2c004d8366bcc22927d68994'
station_id = 'fe67563f37772e63c74487be'
station_ids = [station_id, 'fe35e6509703baebf294c59e']

dataset_id = '2c004d8366bcc22927d68994'
station_id = '673d6d9fca3ccf38fa009ad1'

dataset_id = 'f27574a7b38eab5b0bc2b3d7' # envlib
station_id = 'fedeb59e6c7f47597a7d47c7'

dataset_id = 'de3bff8e3c3a2ad9200d8684'
station_id = 'fedeb59e6c7f47597a7d47c7'

dataset_id = '99d5109c61642e55e8a57657'
station_id = 'de3f6e8951378d6c16186b8f'

dataset_id = 'f27574a7b38eab5b0bc2b3d7'
station_id = '9c90243e84b8c5b17f0726c4'

dataset_id = '0a1583c61202c6791ae39e63'
station_id = '5d06c5a8065a26b51c19b241'

dataset_id = '0a1583c61202c6791ae39e63'

dataset_id = '692e8696cc9e11b4a1ef943d'

dataset_id = '799329b9023c9d9980abd1f6'
station_id = '937f2bb85e82347746fe1be9'

dataset_id = '4f5945540c2391967b550cc6'
station_id = '4db28a9db0cb036507490887'

dataset_id = '52e2196ce75eba1b79e61680'
station_id = '7ddd130ed4b89381d879e0d5'
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

station_ids = [s['station_id'] for s in stn_list1[111:160]]
data2 = self.get_bulk_results(dataset_id, station_ids, output='Dataset', threads=10)

station_ids = [s['station_id'] for s in stn_list1[100:130]]
data3 = self.get_bulk_results(dataset_id, station_ids, output='Dataset', threads=10)

# dataset_id = 'f4cfb5a362707785dd39ff85'
# station_id = 'ff4213c61878e098e07df513'

query_geometry = {'type': 'Point', 'coordinates': [171.043868, -43.106372]}
geom_query = shape(query_geometry).buffer(0.1)

stn_list2 = self.get_stations(dataset_id, query_geometry)
stn_list2 = self.get_stations(dataset_id, lat=-43.1, lon=171.1)
data2 = self.get_results(dataset_id, geometry=query_geometry, output='Dataset')

stn = [s for s in stn_list1 if s['station_id'] == station_id]

stn = [s for s in stn_list1 if 'Waiau River' in s['ref']]


gwl_ds1 = [d for d in self.datasets if d['parameter'] == 'groundwater_depth']
era5_ds1 = [d for d in self.datasets if d['owner'] == 'ECMWF']
nz_ds1 = [d for d in self.datasets if d['owner'] == 'NZ Open Modelling Consortium']
hh_ds1 = [d for d in self.datasets if d['owner'] == 'Headwaters Hydrology'][0]

stns = self.get_stations(hh_ds1['dataset_id'])

stn_id = 'fd60b5bafe19d6b243dda43d'

data1 = self.get_results(hh_ds1['dataset_id'], stn_id, output='Dataset')


type1 = stns[0]['geometry']['type']

if type1 == 'Polygon':
    geo = [s['geometry']['coordinates'][0][0][-1] for s in stns]
    geo1 = np.array(geo).round(5)
    geo1.sort()
    np.diff(geo1)


# geo1 = []
# for s in stn_list1:
#     s1 = {'station_id': s['station_id'], 'lon': s['geometry']['coordinates'][0], 'lat': s['geometry']['coordinates'][1]}
#     geo1.append(s1)

geo1 = []
for s in stn_list1:
    s1 = {'station_id': s['station_id'], 'geo': shape(s['geometry']).simplify(0.1)}
    geo1.append(s1)

sdf = pd.DataFrame(geo1)

sdf2 = gpd.GeoDataFrame(sdf[['station_id']], geometry=sdf['geo'])
sdf2['x'] = sdf2.geometry.x.round(1)
sdf2['y'] = sdf2.geometry.y.round(1)

#####################################################
## dev testing

geometry: dict = None
lat: float = None
lon: float = None
from_date = None
to_date = None
from_mod_date = None
to_mod_date = None
version_date = None
heights = None
bands: int = None
squeeze_dims: bool = False
output: str = 'Dataset'
threads: int = 20

remote = {'bucket': 'typhon', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 4}
remote = {'bucket': 'nz-open-modelling-consortium', 'public_url': 'https://b2.nzrivers.xyz/file/', 'version': 4}
remote = {'bucket': 'fire-emergency-nz', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 4}
remote = {'bucket': 'fire-emergency-nz', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 2}
remote = {'bucket': 'nasa-data', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 4}
remote = {'bucket': 'point-forecasts', 'public_url': 'https://b2.tethys-ts.xyz/file', 'version': 4}
remote = {'bucket': 'ecan-env-monitoring', 'public_url': 'https://b2.tethys-ts.xyz/file', 'version': 4}
remote = {'bucket': 'es-hilltop', 'public_url': 'https://b2.tethys-ts.xyz/file', 'version': 4}

cache = '/media/nvme1/cache/tethys'

dataset_id = '752ee66d969cc09a16efebc7'
station_ids = '80ede07567c4d7cdd00b0954'

dataset_id = 'bb20b3ef3dd4341ee30a2bf0'

dataset_id = '870e79441964b26f0908f732'
station_ids = '8da1e6b2869430ab5aadc0e5'

dataset_id = 'dddb02cd5cb7ae191311ab19'
station_ids = '71369f685f7a5841a060a171'

dataset_id = '0b2bd62cc42f3096136f11e9'
station_ids = 'c8db6013a9eb76705b5c80f2'
ref = 'ashley'

station_ids = 'c15ce95a56b39b6dfeea00e8'

dataset_id = '0de7cbfe05aebc2272ceba17'

dataset_id = 'f56892eb59d12cfbc02acceb'

dataset_id = 'c3a09c8a5da175897916e8e8'

dataset_id = '799329b9023c9d9980abd1f6'

dataset_id = '8d4afa6c8e82d91b81879c12'

dataset_id = 'c3a09c8a5da175897916e8e8'

dataset_id = '63a6144a796c05fc67813d46'

dataset_id = '6c848076f9825bf0eb59a4f2'

self = Tethys([remote], cache=cache)
self = Tethys([remote])
self = Tethys()

rv1 = self.get_versions(dataset_id)
stns1 = self.get_stations(dataset_id)

station_ids = [s['station_id'] for s in stns1[:1]]
station_ids = [s['station_id'] for s in stns1 if ref in s['ref']]

results1 = self.get_results(dataset_id, station_ids, heights=None)

results1 = self.get_results(dataset_id, station_ids, heights=[10])

results1 = self.get_results(dataset_id, station_ids, heights=[10], from_date='2021-04-01')

results1 = self.get_results(dataset_id, station_ids, heights=None, from_date='2021-04-01')


for d in self.datasets:
    rv1 = self.get_versions(d['dataset_id'])
    print(rv1)


results1 = self.get_results(dataset_id, 'fac7e3f6ee48113ccb30e446', heights=None)

results2 = self.get_results(dataset_id, station_ids, heights=[10], version_date='2022-03-01')


station_ids = [s['station_id'] for s in stns1[:3]]

for s in stns1:
    print(s['station_id'])
    out1 = self.get_results(dataset_id, s['station_id'])



### Tests
remote1 = {'bucket': 'fire-emergency-nz', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 4}
remote2 = {'bucket': 'fire-emergency-nz', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 2}

ds_ids = ['c5f55d97e71e7cd73295ad7f', 'dddb02cd5cb7ae191311ab19', '0b2bd62cc42f3096136f11e9']

t1 = Tethys([remote1])
t2 = Tethys([remote2])

for ds_id in ds_ids:
    print('-- ds_id: ' + ds_id)
    stns = t1.get_stations(ds_id)

    stns0 = t2.get_stations(ds_id)

    for stn in stns:
        stn_id = stn['station_id']
        print('-- stn_id: ' + stn_id)
        base_ref = stn['ref']

        if base_ref[-3] == '_':
            old_ref = base_ref[:-3]
        else:
            old_ref = base_ref

        stn0 = [s for s in stns0 if old_ref in s['ref']]

        if stn0:
            stn0 = stn0[0]
            old_stn_id = stn0['station_id']

            r1 = t1.get_results(ds_id, stn_id)
            r1_prod = int(np.prod(list(dict(r1.dims).values())))

            r2 = t2.get_results(ds_id, old_stn_id)
            r2_prod = int(np.prod(list(dict(r2.dims).values())))

            if r1_prod < r2_prod:
                print('{ds_id}: old stn {old_stn_id}: {old_prod}, new stn {new_stn_id}: {new_prod}'.format(ds_id=ds_id, old_stn_id=old_stn_id, new_stn_id=stn_id, old_prod=r2_prod, new_prod=r1_prod))








test_stn_dict = {'c5f55d97e71e7cd73295ad7f': [['91d7d738684b3b6d8e11acd9', '276a3e6428a700229b437626'], ['17c7c90057683b807ad77b10', '751946ea52d04e67639fea1c']],
                 'dddb02cd5cb7ae191311ab19': [['91d7d738684b3b6d8e11acd9', '276a3e6428a700229b437626'], ['17c7c90057683b807ad77b10', '751946ea52d04e67639fea1c']],
                 '0b2bd62cc42f3096136f11e9': [['91d7d738684b3b6d8e11acd9', '276a3e6428a700229b437626'], ['71cd89d47beb79712903eb10', '71369f685f7a5841a060a171'], ['17c7c90057683b807ad77b10', '751946ea52d04e67639fea1c']]}


ds_id = 'c5f55d97e71e7cd73295ad7f'


stn = [s for s in stns if s['station_id'] == '751946ea52d04e67639fea1c'][0]
stn0 = [s for s in stns0 if s['station_id'] == '17c7c90057683b807ad77b10'][0]

[s for s in stns1 if 'Waimak' in s['name']]

[s for s in stns0 if 'poroporo' in s['ref']]















