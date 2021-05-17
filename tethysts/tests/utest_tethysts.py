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

# remote = {'bucket': 'fire-emergency-nz', 'connection_config': 'https://b2.tethys-ts.xyz'}
remote = {'bucket': 'nasa-data', 'connection_config': 'https://b2.tethys-ts.xyz'}
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
# dataset_id='320d6836250169a5f7b78163'
# station_id='7df0d7fe8c6fcd06c50d73a6'
station_id = 'de7cfd749e24a6b78c2281fb'
station_ids = [station_id, 'de7cfd749e24a6b78c2281fb']
dataset_id = '4690821a39b3fb65c197a540'
station_id = '78b7a3d3f61cf33e681490c1'
station_ids = [station_id, '9f3ee54af239b845241f8f13']
dataset_id = '38138ea1c3350031d1b217f6'
station_id = 'b318207aa246e7bbbd74cb19'

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
# data1 = self.get_results(dataset_id, station_id, from_date='2012-01-02 00:00', output='Dataset')

data1 = self.get_results(dataset_id, station_id, output='DataArray')
data2 = self.get_results(dataset_id, station_id2, output='DataArray')


data2 = self.get_bulk_results(dataset_id, station_ids, output='Dataset')

# dataset_id = 'f4cfb5a362707785dd39ff85'
# station_id = 'ff4213c61878e098e07df513'

query_geometry = {'type': 'Point', 'coordinates': [171.043868, -43.106372]}
geom_query = shape(query_geometry).buffer(0.1)

stn_list2 = self.get_stations(dataset_id, query_geometry)
stn_list2 = self.get_stations(dataset_id, lat=-43.1, lon=171.1)
data2 = self.get_results(dataset_id, geometry=query_geometry, squeeze_dims=True, output='Dataset')

stn = [s for s in stn_list1 if s['station_id'] == station_id]


geom0 = [s['geometry'] for s in stn_list1]
geom1 = [shape(s) for s in geom0]

geom2 = [wkb.dumps(s) for s in geom1]
geom3 = [wkb.loads(s) for s in geom2]

geom2 = [s.wkt for s in geom1]
geom3 = [wkt.loads(s) for s in geom2]


strtree = STRtree(geom1)

with open('strtree1.pickle', 'wb') as f:
    pickle.dump(strtree, f, pickle.HIGHEST_PROTOCOL)

with open('strtree1.pickle', 'rb') as f:
    strtree1 = pickle.load(f)

with open('geo1.pickle', 'wb') as f:
    pickle.dump(geom2, f, pickle.HIGHEST_PROTOCOL)

with open('geo2.pickle', 'wb') as f:
    pickle.dump(geom0, f, pickle.HIGHEST_PROTOCOL)



geo2 = {'dataset_id': dataset_id,
        'station_id': [s['station_id'] for s in stn_list1] * 345,
        'geometry': geom0 * 345
        }

with open('geo3.pickle', 'wb') as f:
    pickle.dump(geo2, f, pickle.HIGHEST_PROTOCOL)


json1 = orjson.dumps(geo2, option=orjson.OPT_SERIALIZE_NUMPY)

gb1 = tu.write_json_zstd(geo2)

geo3 = tu.read_json_zstd(gb1)

geo4 = [shape(s) for s in geo3['geometry']]

strtree = STRtree(geo4)

gb2 = tu.write_json_zstd(stn_list1)


geom_list = [s['geometry'] for s in stn_list1]
stns = stn_list1.copy()

query_geometry = {'type': 'Point', 'coordinates': [171.1, -43.1]}
geom_query = shape(query_geometry).buffer(0.1)


def get_nearest_station(stns, geom_query):
    """

    """
    if isinstance(geom_query, dict):
        geom_query = shape(geom_query)

    geom1 = [shape(s['geometry']) for s in stns]
    strtree = STRtree(geom1)
    res = strtree.nearest(geom_query)
    res_id = res.wkb_hex

    stn_id_dict = {shape(s['geometry']).wkb_hex: s['station_id'] for s in stns}

    stn_id = stn_id_dict[res_id]

    return stn_id


def get_intersected_stations(stns, geom_query):
    """

    """
    if isinstance(geom_query, dict):
        geom_query = shape(geom_query)

    geom1 = [shape(s['geometry']) for s in stns]
    strtree = STRtree(geom1)
    res = strtree.query(geom_query)
    res_ids = [r.wkb_hex for r in res]

    stn_id_dict = {shape(s['geometry']).wkb_hex: s['station_id'] for s in stns}

    stn_ids = [stn_id_dict[r] for r in res_ids]

    return stn_ids




























