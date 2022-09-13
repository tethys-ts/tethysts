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
import tethys_data_models as tdm
import pathlib
from functools import partial
from pydantic import HttpUrl
import shutil
import gzip
# import psutil

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
    ## Stations filter
    rc2 = copy.deepcopy([rc for rc in results_chunks if rc['station_id'] in stn_ids])

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

    if len(rc2) == 0:
        return rc2

    ## Sort by mod date
    rc2.sort(key=lambda d: d['modified_date'] if 'modified_date' in d else '1900-01-01')

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


def process_results_output(ts_xr, output='xarray', squeeze_dims=False):
    """

    """
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


def load_dataset(results, from_date=None, to_date=None):
    """

    """
    if isinstance(results, (pathlib.Path, str)):
        data = xr.open_dataset(results)
    elif isinstance(results, bytes):
        try:
            data = xr.load_dataset(read_pkl_zstd(results))
        except:
            data = xr.load_dataset(results)
    elif isinstance(results, xr.Dataset):
        data = results
    else:
        raise TypeError('Not the right data type.')

    chunk_vars = [v for v in list(data.variables) if ('chunk' in v)]
    data = data.drop_vars(chunk_vars)

    # if 'geometry' in data.dims:
    #     stn_vars = [v for v in list(data.data_vars) if ('time' not in data[v].dims) and (v not in ['station_id', 'lon', 'lat'])]
    #     data = data.drop_vars(stn_vars)

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

    return data


def download_results(chunk: dict, bucket: str, s3: botocore.client.BaseClient = None, connection_config: dict = None, public_url: HttpUrl = None, cache: Union[pathlib.Path] = None, from_date=None, to_date=None, return_raw=False):
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
                data_obj = get_object_s3(chunk['key'], bucket, s3, connection_config, public_url)
                with open(chunk_path, 'wb') as f:
                    f.write(data_obj)

        data_obj = chunk_path

    else:
        data_obj = get_object_s3(chunk['key'], bucket, s3, connection_config, public_url)

        if return_raw:
            return data_obj

    data = load_dataset(data_obj, from_date=from_date, to_date=to_date)

    return data


def xr_concat(datasets: List[xr.Dataset]):
    """
    A much more efficient concat/combine of xarray datasets. It's also much safer on memory.
    """
    # Get variables for the creation of blank dataset
    coords_list = []
    chunk_dict = {}

    for chunk in datasets:
        coords_list.append(chunk.coords.to_dataset())
        for var in chunk.data_vars:
            if var not in chunk_dict:
                dims = tuple(chunk[var].dims)
                enc = chunk[var].encoding.copy()
                dtype = chunk[var].dtype
                _ = [enc.pop(d) for d in ['original_shape', 'source'] if d in enc]
                var_dict = {'dims': dims, 'enc': enc, 'dtype': dtype, 'attrs': chunk[var].attrs}
                chunk_dict[var] = var_dict

    try:
        xr3 = xr.combine_by_coords(coords_list, compat='override', data_vars='minimal', coords='all', combine_attrs='override')
    except:
        xr3 = xr.merge(coords_list, compat='override', combine_attrs='override')

    # Run checks - requires psutil which I don't want to make it a dep yet...
    # available_memory = getattr(psutil.virtual_memory(), 'available')
    # dims_dict = dict(xr3.coords.dims)
    # size = 0
    # for var, var_dict in chunk_dict.items():
    #     dims = var_dict['dims']
    #     dtype_size = var_dict['dtype'].itemsize
    #     n_dims = np.prod([dims_dict[dim] for dim in dims])
    #     size = size + (n_dims*dtype_size)

    # if size >= available_memory:
    #     raise MemoryError('Trying to create a dataset of size {}MB, while there is only {}MB available.'.format(int(size*10**-6), int(available_memory*10**-6)))

    # Create the blank dataset
    for var, var_dict in chunk_dict.items():
        dims = var_dict['dims']
        shape = tuple(xr3[c].shape[0] for c in dims)
        xr3[var] = (dims, np.full(shape, np.nan, var_dict['dtype']))
        xr3[var].attrs = var_dict['attrs']
        xr3[var].encoding = var_dict['enc']

    # Fill the dataset with data
    for chunk in datasets:
        for var in chunk.data_vars:
            if isinstance(chunk[var].variable._data, np.ndarray):
                xr3[var].loc[chunk[var].transpose(*chunk_dict[var]['dims']).coords.indexes] = chunk[var].transpose(*chunk_dict[var]['dims']).values
            elif isinstance(chunk[var].variable._data, xr.core.indexing.MemoryCachedArray):
                c1 = chunk[var].copy().load().transpose(*chunk_dict[var]['dims'])
                xr3[var].loc[c1.coords.indexes] = c1.values
                c1.close()
                del c1
            else:
                raise TypeError('Dataset data should be either an ndarray or a MemoryCachedArray.')

    return xr3

























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



























































