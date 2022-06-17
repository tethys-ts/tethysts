"""


"""
from io import BytesIO, SEEK_SET, SEEK_END, DEFAULT_BUFFER_SIZE
import os
import numpy as np
import requests
import xarray as xr
import pandas as pd
import orjson
from datetime import datetime
import zstandard as zstd
import pickle
import copy
import boto3
import botocore
from time import sleep
from shapely.geometry import shape, Polygon, Point
from shapely.strtree import STRtree
from typing import Optional, List, Any, Union
from scipy import spatial
import traceback
import tethys_data_models as tdm
import pathlib
from functools import partial
from pydantic import HttpUrl
import shutil
import gzip

pd.options.display.max_columns = 10

##############################################
### Reference objects

b2_public_key_pattern = '{base_url}/{bucket}/{obj_key}'
contabo_public_key_pattern = '{base_url}:{bucket}/{obj_key}'
public_remote_key = 'https://b2.tethys-ts.xyz/file/tethysts/tethys/public_remotes_v4.json.zst'

local_results_name = '{ds_id}/{stn_id}/{chunk_id}.{version_date}.{chunk_hash}.nc'

##############################################
### Helper functions


def update_nested(in_dict, ds_id, version_date, value):
    """

    """
    if ds_id in in_dict:
        in_dict[ds_id][version_date] = value
    else:
        in_dict.update({ds_id: {version_date: value}})


def make_run_date_key(run_date=None):
    """

    """
    if run_date is None:
        run_date = pd.Timestamp.today(tz='utc')
        run_date_key = run_date.strftime('%Y%m%dT%H%M%SZ')
    elif isinstance(run_date, pd.Timestamp):
        run_date_key = run_date.strftime('%Y%m%dT%H%M%SZ')
    elif isinstance(run_date, str):
        run_date1 = pd.Timestamp(run_date).tz_localize(None)
        run_date_key = run_date1.strftime('%Y%m%dT%H%M%SZ')
    else:
        raise TypeError('run_date must be None, Timestamp, or a string representation of a timestamp')

    return run_date_key


def create_public_s3_url(base_url, bucket, obj_key):
    """
    This should be updated as more S3 providers are added!
    """
    if 'contabo' in base_url:
        key = contabo_public_key_pattern.format(base_url=base_url.rstrip('/'), bucket=bucket, obj_key=obj_key)
    else:
        key = b2_public_key_pattern.format(base_url=base_url.rstrip('/'), bucket=bucket, obj_key=obj_key)

    return key


class ResponseStream(object):
    """
    In many applications, you'd like to access a requests response as a file-like object, simply having .read(), .seek(), and .tell() as normal. Especially when you only want to partially download a file, it'd be extra convenient if you could use a normal file interface for it, loading as needed.

This is a wrapper class for doing that. Only bytes you request will be loaded - see the example in the gist itself.

https://gist.github.com/obskyr/b9d4b4223e7eaf4eedcd9defabb34f13
    """
    def __init__(self, request_iterator):
        self._bytes = BytesIO()
        self._iterator = request_iterator

    def _load_all(self):
        self._bytes.seek(0, SEEK_END)
        for chunk in self._iterator:
            self._bytes.write(chunk)

    def _load_until(self, goal_position):
        current_position = self._bytes.seek(0, SEEK_END)
        while current_position < goal_position:
            try:
                current_position += self._bytes.write(next(self._iterator))
            except StopIteration:
                break

    def tell(self):
        return self._bytes.tell()

    def read(self, size=None):
        left_off_at = self._bytes.tell()
        if size is None:
            self._load_all()
        else:
            goal_position = left_off_at + size
            self._load_until(goal_position)

        self._bytes.seek(left_off_at)
        return self._bytes.read(size)

    def seek(self, position, whence=SEEK_SET):
        if whence == SEEK_END:
            self._load_all()
        else:
            self._bytes.seek(position, whence)


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def get_nearest_station(stns, geom_query):
    """

    """
    if isinstance(geom_query, dict):
        geom_query = shape(geom_query)

    geom1 = [shape(s['geometry']) for i, s in stns.items()]
    strtree = STRtree(geom1)
    res = strtree.nearest(geom_query)
    res_id = res.wkb_hex

    stn_id_dict = {shape(s['geometry']).wkb_hex: i for i, s in stns.items()}

    stn_id = stn_id_dict[res_id]

    return stn_id


def get_intersected_stations(stns, geom_query):
    """

    """
    if isinstance(geom_query, dict):
        geom_query = shape(geom_query)

    geom1 = [shape(s['geometry']) for i, s in stns.items()]
    strtree = STRtree(geom1)
    res = strtree.query(geom_query)
    res_ids = [r.wkb_hex for r in res]

    stn_id_dict = {shape(s['geometry']).wkb_hex: i for i, s in stns.items()}

    stn_ids = [stn_id_dict[r] for r in res_ids]

    return stn_ids


def spatial_query(stns: dict,
                  query_geometry: Optional[dict] = None,
                  lat: Optional[float] = None,
                  lon: Optional[float] = None,
                  distance: Optional[float] = None):
    """

    """
    if isinstance(lat, float) and isinstance(lon, float):
        geom_query = Point(lon, lat)
        if isinstance(distance, (int, float)):
            geom_query = geom_query.buffer(distance)
            stn_ids = get_intersected_stations(stns, geom_query)
        else:
            stn_ids = [get_nearest_station(stns, geom_query)]
    elif isinstance(query_geometry, dict):
        geom_query = shape(query_geometry)
        if isinstance(geom_query, Point):
            stn_ids = [get_nearest_station(stns, geom_query)]
        elif isinstance(geom_query, Polygon):
            stn_ids = get_intersected_stations(stns, geom_query)
        else:
            raise ValueError('query_geometry must be a Point or Polygon dict.')
    else:
        stn_ids = None

    return stn_ids


def get_nearest_from_extent(data,
                            query_geometry: Optional[dict] = None,
                            lat: Optional[float] = None,
                            lon: Optional[float] = None):
    """

    """
    ## Prep the query point
    if isinstance(lat, float) and isinstance(lon, float):
        geom_query = Point(lon, lat)
    elif isinstance(query_geometry, dict):
        geom_query = shape(query_geometry)
        if not isinstance(geom_query, Point):
            raise ValueError('query_geometry must be a Point.')
    else:
        raise ValueError('query_geometry or lat/lon must be passed as a Point.')

    ## Prep the input data
    if 'geometry' in data:
        raise NotImplementedError('Need to implement geometry blocks nearest query.')
    else:
        lats = data['lat'].values
        lons = data['lon'].values
        xy = cartesian_product(lons, lats)
        kdtree = spatial.cKDTree(xy)
        dist, index = kdtree.query(geom_query.coords[0])
        lon_n, lat_n = xy[index]

    data1 = data.sel(lon=[lon_n], lat=[lat_n])

    return data1


def read_pkl_zstd(obj, unpickle=False):
    """
    Deserializer from a pickled object compressed with zstandard.

    Parameters
    ----------
    obj : bytes or str
        Either a bytes object that has been pickled and compressed or a str path to the file object.
    unpickle : bool
        Should the bytes object be unpickled or left as bytes?

    Returns
    -------
    Python object
    """
    if isinstance(obj, str):
        with open(obj, 'rb') as p:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(p) as reader:
                obj1 = reader.read()

    elif isinstance(obj, bytes):
        dctx = zstd.ZstdDecompressor()
        obj1 = dctx.decompress(obj)
    else:
        raise TypeError('obj must either be a str path or a bytes object')

    if unpickle:
        obj1 = pickle.loads(obj1)

    return obj1


def read_json_zstd(obj):
    """
    Deserializer from a compressed zstandard json object to a dictionary.

    Parameters
    ----------
    obj : bytes
        The bytes object.

    Returns
    -------
    Dict
    """
    if isinstance(obj, str):
        with open(obj, 'rb') as p:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(p) as reader:
                obj1 = reader.read()
    elif isinstance(obj, bytes):
        dctx = zstd.ZstdDecompressor()
        obj1 = dctx.decompress(obj)
    else:
        raise TypeError('obj must either be a str path or a bytes object')

    dict1 = orjson.loads(obj1)

    return dict1


def s3_client(connection_config: dict, max_pool_connections: int = 30):
    """
    Function to establish a client connection with an S3 account. This can use the legacy connect (signature_version s3) and the curent version.

    Parameters
    ----------
    connection_config : dict
        A dictionary of the connection info necessary to establish an S3 connection. It should contain service_name, endpoint_url, aws_access_key_id, and aws_secret_access_key. connection_config can also be a URL to a public S3 bucket.
    max_pool_connections : int
        The number of simultaneous connections for the S3 connection.

    Returns
    -------
    S3 client object
    """
    ## Validate config
    _ = tdm.base.ConnectionConfig(**connection_config)

    s3_config = copy.deepcopy(connection_config)

    if 'config' in s3_config:
        config0 = s3_config.pop('config')
        config0.update({'max_pool_connections': max_pool_connections})
        config1 = boto3.session.Config(**config0)

        s3_config1 = s3_config.copy()
        s3_config1.update({'config': config1})

        s3 = boto3.client(**s3_config1)
    else:
        s3_config.update({'config': botocore.config.Config(max_pool_connections=max_pool_connections)})
        s3 = boto3.client(**s3_config)

    return s3


def get_object_s3(obj_key: str, bucket: str, s3: botocore.client.BaseClient = None, connection_config: dict = None, public_url: HttpUrl=None, version_id=None, range_start: int=None, range_end: int=None, counter=5):
    """
    General function to get an object from an S3 bucket. One of s3, connection_config, or public_url must be used.

    Parameters
    ----------
    obj_key : str
        The object key in the S3 bucket.
    s3 : botocore.client.BaseClient
        An S3 object created via the s3_client function.
    connection_config : dict
        A dictionary of the connection info necessary to establish an S3 connection. It should contain service_name, s3, endpoint_url, aws_access_key_id, and aws_secret_access_key.
    public_url : str
        A URL to a public S3 bucket. This is generally only used for Backblaze object storage.
    bucket : str
        The bucket name.
    counter : int
        Number of times to retry to get the object.

    Returns
    -------
    bytes
        bytes object of the S3 object.
    """
    counter1 = counter

    get_dict = {'Key': obj_key, 'Bucket': bucket}

    if isinstance(version_id, str):
        get_dict['VersionId'] = version_id

    ## Range
    range_dict = {}

    if range_start is not None:
        range_dict['start'] = str(range_start)
    else:
        range_dict['start'] = ''

    if range_end is not None:
        range_dict['end'] = str(range_end)
    else:
        range_dict['end'] = ''

    if range_dict:
        get_dict['Range'] = 'bytes={start}-{end}'.format(**range_dict)

    ## Get the object
    while True:
        try:
            if isinstance(public_url, str) and (version_id is None):
                url = create_public_s3_url(public_url, bucket, obj_key)
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()

                ts_obj = resp.content

            elif isinstance(s3, botocore.client.BaseClient):
                ts_resp = s3.get_object(**get_dict)
                ts_obj = ts_resp.pop('Body').read()

            elif isinstance(connection_config, dict):
                ## Validate config
                _ = tdm.base.ConnectionConfig(**connection_config)

                s3 = s3_client(connection_config)

                ts_resp = s3.get_object(**get_dict)
                ts_obj = ts_resp.pop('Body').read()
            else:
                raise TypeError('One of s3, connection_config, or public_url needs to be correctly defined.')
            break
        except:
            # print(traceback.format_exc())
            if counter1 == 0:
                # raise ValueError('Could not properly download the object after several tries')
                print('Object could not be downloaded.')
                return None
            else:
                # print('Could not properly extract the object; trying again in 5 seconds')
                counter1 = counter1 - 1
                sleep(5)

    return ts_obj


def chunk_filters(results_chunks, stn_ids, time_interval=None, from_date=None, to_date=None, heights=None, bands=None):
    """

    """
    rc2 = copy.deepcopy(results_chunks)

    ## Stations filter
    rc2 = [rc for rc in rc2 if rc['station_id'] in stn_ids]

    ## Temporal filter
    if isinstance(from_date, (str, pd.Timestamp, datetime)) and ('chunk_day' in rc2[0]):
        from_date1 = int(pd.Timestamp(from_date).timestamp()/60/60/24)
        rc2 = [rc for rc in rc2 if (rc['chunk_day'] + time_interval) >= from_date1]

    if len(rc2) == 0:
        return rc2

    if isinstance(to_date, (str, pd.Timestamp, datetime)) and ('chunk_day' in rc2[0]):
        to_date1 = int(pd.Timestamp(to_date).timestamp()/60/60/24)
        rc2 = [rc for rc in rc2 if rc['chunk_day'] <= to_date1]

    if len(rc2) == 0:
        return rc2

    ## Heights and bands filter
    if (heights is not None) and ('height' in rc2[0]):
        if isinstance(heights, (int, float)):
            h1 = [int(heights*1000)]
        elif isinstance(heights, list):
            h1 = [int(h*1000) for h in heights]
        else:
            raise TypeError('heights must be an int, float, or list of int/float.')
        rc2 = [rc for rc in rc2 if rc['height'] in h1]

    if len(rc2) == 0:
        return rc2

    if (bands is not None) and ('band' in rc2[0]):
        if isinstance(bands, int):
            b1 = [heights]
        elif isinstance(bands, list):
            b1 = [int(b) for b in bands]
        else:
            raise TypeError('bands must be an int or list of int.')
        rc2 = [rc for rc in rc2 if rc['band'] in b1]

    return rc2


def result_filters(ts_xr, from_date=None, to_date=None, from_mod_date=None, to_mod_date=None):
    """

    """
    if isinstance(from_date, (str, pd.Timestamp, datetime)):
        from_date1 = pd.Timestamp(from_date)
    else:
        from_date1 = None
    if isinstance(to_date, (str, pd.Timestamp, datetime)):
        to_date1 = pd.Timestamp(to_date)
    else:
        to_date1 = None

    if isinstance(from_mod_date, (str, pd.Timestamp, datetime)):
        from_mod_date1 = pd.Timestamp(from_mod_date)
    else:
        from_mod_date1 = None
    if isinstance(to_mod_date, (str, pd.Timestamp, datetime)):
        to_mod_date1 = pd.Timestamp(to_mod_date)
    else:
        to_mod_date1 = None

    if (to_date1 is not None) or (from_date1 is not None):
        ts_xr1 = ts_xr.sel(time=slice(from_date1, to_date1))
    else:
        ts_xr1 = ts_xr

    if (to_mod_date1 is not None) or (from_mod_date1 is not None):
        if 'modified_date' in ts_xr1:
            ts_xr1 = ts_xr1.sel(modified_date=slice(from_mod_date1, to_mod_date1))

    return ts_xr1


def process_results_output(ts_xr, parameter, modified_date=False, quality_code=False, output='xarray', squeeze_dims=False,
                           # include_chunk_vars: bool = False
                           ):
    """

    """
    out_param = [parameter]

    if quality_code:
        if 'quality_code' in ts_xr:
            out_param.extend(['quality_code'])

    if modified_date:
        if 'modified_date' in ts_xr:
            out_param.extend(['modified_date'])

    if len(out_param) == 1:
        out_param = out_param[0]

    # if not include_chunk_vars:
    #     chunk_vars = [v for v in list(ts_xr.variables) if 'chunk' in v]
    #     ts_xr = ts_xr.drop(chunk_vars)

    ## Return
    if squeeze_dims:
        ts_xr = ts_xr.squeeze()

    if output == 'xarray':
        return ts_xr

    elif output == 'dict':
        data_dict = ts_xr.to_dict()

        return data_dict

    elif output == 'json':
        json1 = orjson.dumps(ts_xr.to_dict(), option=orjson.OPT_OMIT_MICROSECONDS | orjson.OPT_SERIALIZE_NUMPY)

        return json1
    else:
        raise ValueError("output must be one of 'xarray', 'dict', or 'json'")


# def convert_results_v2_to_v3(data):
#     """
#     Function to convert xarray Dataset results in verion 2 structure to version 3 structure.
#     """
#     geo1 = Point(float(data['lon']), float(data['lat'])).wkb_hex
#     data2 = data.assign_coords({'geometry': geo1})
#     if 'virtual_station' in data2:
#         data2 = data2.drop_vars('virtual_station')

#     # data2['station_id'].attrs = data['station_id'].attrs
#     data2['geometry'].attrs = {'long_name': 'The hexadecimal encoding of the Well-Known Binary (WKB) geometry', 'crs_EPSG': 4326}

#     data2 = data2.expand_dims('geometry')
#     # data2 = data2.expand_dims('height')

#     # vars1 = list(data2.variables)
#     # param = [p for p in vars1 if 'dataset_id' in data2[p].attrs][0]
#     # param_attrs = data2[param].attrs

#     # if 'result_type' in data2[param].attrs:
#     #     _ = data2[param].attrs.pop('result_type')
#     # data2[param].attrs.update({'spatial_distribution': 'sparse', 'geometry_type': 'Point', 'grouping': 'none'})

#     # params = [param]
#     # if 'ancillary_variables' in param_attrs:
#     #     params.extend(param_attrs['ancillary_variables'].split(' '))

#     # for p in params:
#     #     data2[p] = data2[p].expand_dims('height')

#     data2.attrs.update({'version': 3})

#     return data2


# def convert_results_v3_to_v4(data):
#     """
#     Function to convert xarray Dataset results in verion 3 structure to version 4 structure.
#     """
#     ## Change the extent to station_geometry
#     if 'extent' in list(data.coords):
#         data = data.rename({'extent': 'station_geometry'})

#     ## Change spatial_distribution to result_type
#     vars1 = list(data.variables)
#     param = [p for p in vars1 if 'dataset_id' in data[p].attrs][0]
#     param_attrs = data[param].attrs

#     if 'result_type' in param_attrs:
#         _ = data[param].attrs.pop('result_type')
#         data[param].attrs.update({'spatial_distribution': 'sparse', 'geometry_type': 'Point', 'grouping': 'none'})

#     sd_attr = param_attrs.pop('spatial_distribution')

#     if sd_attr == 'sparse':
#         result_type = 'time_series'
#     else:
#         result_type = 'grid'

#     data[param].attrs.update({'result_type': result_type})

#     ## change base attrs
#     _ = data.attrs.pop('featureType')
#     data.attrs.update({'result_type': result_type, 'version': 4})

#     return data


# def read_in_chunks(file_object, chunk_size=524288):
#     while True:
#         data = file_object.read(chunk_size)
#         if not data:
#             break
#         yield data


def local_file_byte_iterator(path, chunk_size=DEFAULT_BUFFER_SIZE):
    """given a path, return an iterator over the file
    that lazily loads the file.
    https://stackoverflow.com/a/37222446/6952674
    """
    path = pathlib.Path(path)
    with path.open('rb') as file:
        reader = partial(file.read1, DEFAULT_BUFFER_SIZE)
        file_iterator = iter(reader, bytes())
        for chunk in file_iterator:
            yield from chunk


# def file_byte_iterator(file, chunk_size=DEFAULT_BUFFER_SIZE):
#     """given a path, return an iterator over the file
#     that lazily loads the file.
#     https://stackoverflow.com/a/37222446/6952674
#     """
#     reader = partial(file.read, chunk_size)
#     file_iterator = iter(reader, bytes())
#     for chunk in file_iterator:
#         yield from chunk


def url_stream_to_file(url, file_path, compression=None, chunk_size=524288):
    """

    """
    file_path1 = pathlib.Path(file_path)
    if file_path1.is_dir():
        file_name = url.split('/')[-1]
        file_path2 = str(file_path1.joinpath(file_name))
    else:
        file_path2 = file_path

    base_path = os.path.split(file_path2)[0]
    os.makedirs(base_path, exist_ok=True)

    counter = 4
    while True:
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                stream = ResponseStream(r.iter_content(chunk_size))

                if compression == 'zstd':
                    if str(file_path2).endswith('.zst'):
                        file_path2 = os.path.splitext(file_path2)[0]
                    dctx = zstd.ZstdDecompressor()

                    with open(file_path2, 'wb') as f:
                        dctx.copy_stream(stream, f, read_size=chunk_size, write_size=chunk_size)

                elif compression == 'gzip':
                    if str(file_path2).endswith('.gz'):
                        file_path2 = os.path.splitext(file_path2)[0]

                    with gzip.open(stream, 'rb') as s_file, open(file_path2, 'wb') as d_file:
                        shutil.copyfileobj(s_file, d_file, chunk_size)
                else:
                    with open(file_path2, 'wb') as f:
                        chunk = stream.read(chunk_size)
                        while chunk:
                            f.write(chunk)
                            chunk = stream.read(chunk_size)

                break

        except Exception as err:
            if counter < 1:
                raise err
            else:
                counter = counter - 1
                sleep(5)

    return file_path2


def download_results(index: dict, dims: list, chunk: dict, bucket: str, s3: botocore.client.BaseClient = None, connection_config: dict = None, public_url: HttpUrl = None, cache: Union[pathlib.Path] = None, from_date=None, to_date=None):
    """

    """
    if isinstance(cache, pathlib.Path):
        chunk_hash = chunk['chunk_hash']
        version_date = pd.Timestamp(chunk['version_date']).strftime('%Y%m%d%H%M%SZ')
        results_file_name = local_results_name.format(ds_id=chunk['dataset_id'], stn_id=chunk['station_id'], chunk_id=chunk['chunk_id'], version_date=version_date, chunk_hash=chunk_hash)
        chunk_path = cache.joinpath(results_file_name)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)

        if not chunk_path.exists():
            if public_url is not None:
                url = create_public_s3_url(public_url, bucket, chunk['key'])
                _ = url_stream_to_file(url, chunk_path, compression='zstd')
            else:
                obj1 = get_object_s3(chunk['key'], bucket, s3, connection_config, public_url)
                with open(chunk_path, 'wb') as f:
                    f.write(obj1)
                del obj1

        data = xr.open_dataset(chunk_path)

    else:
        obj1 = get_object_s3(chunk['key'], bucket, s3, connection_config, public_url)
        data = xr.load_dataset(read_pkl_zstd(obj1))
        del obj1

    chunk_vars = [v for v in list(data.variables) if ('chunk' in v)]
    data = data.drop_vars(chunk_vars)

    if 'station_geometry' in data.dims:
        stn_vars = [d for d in data.variables if 'station_geometry' in data[d].dims]
        data = data.drop_vars(stn_vars)

    if isinstance(from_date, (str, pd.Timestamp, datetime)):
        from_date1 = pd.Timestamp(from_date)
    else:
        from_date1 = None

    if isinstance(to_date, (str, pd.Timestamp, datetime)):
        to_date1 = pd.Timestamp(to_date)
    else:
        to_date1 = None

    if (to_date1 is not None) or (from_date1 is not None):
        data = data.sel(time=slice(from_date1, to_date1))

    stn_id = chunk['station_id']
    if len(dims) == 1:
        pos0 = chunk[dims[0]+'_pos']
        index[stn_id][pos0] = data
    elif len(dims) == 2:
        pos0 = chunk[dims[0]+'_pos']
        pos1 = chunk[dims[1]+'_pos']
        index[stn_id][pos0][pos1] = data
    elif len(dims) == 3:
        pos0 = chunk[dims[0]+'_pos']
        pos1 = chunk[dims[1]+'_pos']
        pos2 = chunk[dims[2]+'_pos']
        index[stn_id][pos0][pos1][pos2] = data


# def v2_v3_results_chunks(results_obj):
#     """
#     Function to convert version 2 and 3 data into result chunks and result versions. This conversion only keeps the last version of the results.
#     """
#     last_version = max([obj['results_object_key'][-1]['run_date'] for obj in results_obj])

#     results_chunks = []

#     for obj in results_obj:
#         last_obj = obj['results_object_key'][-1]
#         rc1 = {'chunk_id': '',
#                'chunk_hash': '',
#                'dataset_id': obj['dataset_id'],
#                'station_id': obj['station_id'],
#                'content_length': last_obj['content_length'],
#                'key': last_obj['key'],
#                'version_date': pd.Timestamp(last_version)
#                }

#         results_chunks.append(rc1)

#     results_version = [{'dataset_id': obj['dataset_id'],
#                'version_date': last_version,
#                'modified_date': last_version}]

#     return results_version, results_chunks


# def load_dataset(results, from_date=None, to_date=None):
#     """

#     """
#     if isinstance(results, (pathlib.Path, str)):
#         data = xr.load_dataset(results)
#     else:
#         data = results

#     chunk_vars = [v for v in list(data.variables) if ('chunk' in v)]
#     data = data.drop_vars(chunk_vars)

#     if 'station_geometry' in data.dims:
#         stn_vars = [d for d in data.variables if 'station_geometry' in data[d].dims]
#         data = data.drop_vars(stn_vars)

#     if isinstance(from_date, (str, pd.Timestamp, datetime)):
#         from_date1 = pd.Timestamp(from_date)
#     else:
#         from_date1 = None

#     if isinstance(to_date, (str, pd.Timestamp, datetime)):
#         to_date1 = pd.Timestamp(to_date)
#     else:
#         to_date1 = None

#     if (to_date1 is not None) or (from_date1 is not None):
#         data = data.sel(time=slice(from_date1, to_date1))

#     return data


def nest_results(chunks):
    """

    """
    ## Determine all of the dimensions
    dim_order = ['chunk_day', 'height', 'band']
    chunk = chunks[0]
    dims = [dim for dim in chunk if dim in ['chunk_day', 'height', 'band']]
    dims = [dim for dim in dim_order if dim in dims]

    dims_set_dict = {}
    for chunk in chunks:
        stn_id = chunk['station_id']
        if stn_id in dims_set_dict:
            for dim in dims:
                dims_set_dict[stn_id][dim].add(chunk[dim])
        else:
            dict1 = {}
            for dim in dims:
                dict1[dim] = set([chunk[dim]])
            dims_set_dict[stn_id] = dict1

    stn_index = {}
    for stn_id, stn in dims_set_dict.items():
        # stn_index[stn_id] = {}
        dim_shape = []
        for dim in dims:
            len1 = len(stn[dim])
            dim_shape.append(len1)
            dims_set_dict[stn_id][dim] = list(stn[dim])
            dims_set_dict[stn_id][dim].sort()
        stn_index[stn_id] = np.zeros(tuple(dim_shape), dtype=int).tolist()

    for chunk in chunks:
        stn_id = chunk['station_id']
        for dim in dims:
            chunk[dim+'_pos'] = dims_set_dict[stn_id][dim].index(chunk[dim])

    return chunks, stn_index, dims























## Currently not working because of cloudflare CDN
# def get_zstd_url_data(url, request_type='get', headers=None, chunk_size=524288, **kwargs):
#     """

#     """
#     h1 = {'Accept-Encoding': 'zstd'}
#     if isinstance(headers, dict):
#         h1.update(headers)

#     if request_type.lower() == 'get':
#         r1 = requests.get
#     elif request_type.lower() == 'post':
#         r1 = requests.post
#     else:
#         raise ValueError('request_type must be either get or post.')

#     with r1(url, stream=True, timeout=300, headers=h1) as r:
#         r.raise_for_status()
#         stream = ResponseStream(r.iter_content(chunk_size))

#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(stream) as reader:
#             data = reader.read()

#     return data



























































