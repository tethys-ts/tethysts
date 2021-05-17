"""


"""
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
from multiprocessing.pool import ThreadPool
from time import sleep
from shapely.geometry import shape, Polygon, Point
from shapely.strtree import STRtree
from typing import Optional, List, Any, Union
from scipy import spatial
import traceback

pd.options.display.max_columns = 10


##############################################
### Reference objects

key_patterns = {'results': 'tethys/v2/{dataset_id}/{station_id}/{run_date}/results.nc.zst',
                'results_buffer': 'tethys/v2/{dataset_id}/{station_id}/{run_date}/results_buffer.nc.zst',
                'datasets': 'tethys/v2/datasets.json.zst',
                'stations': 'tethys/v2/{dataset_id}/stations.json.zst',
                'station': 'tethys/v2/{dataset_id}/{station_id}/station.json.zst',
                'dataset': 'tethys/v2/{dataset_id}/dataset.json.zst',
                }

b2_public_key_pattern = '{base_url}/file/{bucket}/{obj_key}'

##############################################
### Helper functions


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
        dist, index = kdtree.query(np.array(geom_query))
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
    dctx = zstd.ZstdDecompressor()
    if isinstance(obj, str):
        with open(obj, 'rb') as p:
            obj1 = dctx.decompress(p.read())
    elif isinstance(obj, bytes):
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
    dctx = zstd.ZstdDecompressor()
    obj1 = dctx.decompress(obj)
    dict1 = orjson.loads(obj1)

    return dict1


def s3_connection(connection_config, max_pool_connections=30):
    """
    Function to establish a connection with an S3 account. This can use the legacy connect (signature_version s3) and the curent version.

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


def get_object_s3(obj_key, connection_config, bucket, compression=None, counter=5):
    """
    General function to get an object from an S3 bucket.

    Parameters
    ----------
    obj_key : str
        The object key in the S3 bucket.
    connection_config : dict
        A dictionary of the connection info necessary to establish an S3 connection. It should contain service_name, s3, endpoint_url, aws_access_key_id, and aws_secret_access_key. connection_config can also be a URL to a public S3 bucket.
    bucket : str
        The bucket name.
    compression : None or str
        The compression of the object that should be decompressed. Options include zstd.
    counter : int
        Number of times to retry to get the object.

    Returns
    -------
    bytes
        bytes object of the S3 object.
    """
    counter1 = counter
    while True:
        try:
            if isinstance(connection_config, dict):
                s3 = s3_connection(connection_config)

                ts_resp = s3.get_object(Key=obj_key, Bucket=bucket)
                ts_obj = ts_resp.pop('Body').read()

            elif isinstance(connection_config, str):
                url = b2_public_key_pattern.format(base_url=connection_config, bucket=bucket, obj_key=obj_key)
                resp = requests.get(url)
                resp.raise_for_status()

                ts_obj = resp.content

            if isinstance(compression, str):
                if compression == 'zstd':
                    ts_obj = read_pkl_zstd(ts_obj, False)
                else:
                    raise ValueError('compression option can only be zstd or None')
            break
        except:
            print(traceback.format_exc())
            if counter1 == 0:
                raise ValueError('Could not properly extract the object after several tries')
            else:
                print('Could not properly extract the object; trying again in 5 seconds')
                counter1 = counter1 - 1
                sleep(5)

    return ts_obj


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


def process_results_output(ts_xr, parameter, modified_date=False, quality_code=False, output='DataArray', squeeze_dims=False):
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

    ## Return
    if squeeze_dims:
        ts_xr = ts_xr.squeeze()

    if output == 'Dataset':
        return ts_xr

    elif output == 'DataArray':
        return ts_xr[out_param]

    elif output == 'Dict':
        # darr = ts_xr[out_param]
        darr = ts_xr
        data_dict = darr.to_dict()
        # if 'name' in data_dict:
        #     data_dict.pop('name')

        return data_dict

    elif output == 'json':
        # darr = ts_xr[out_param]
        darr = ts_xr
        data_dict = darr.to_dict()
        # if 'name' in data_dict:
        #     data_dict.pop('name')
        json1 = orjson.dumps(data_dict)

        return json1
    else:
        raise ValueError("output must be one of 'Dataset', 'DataArray', 'Dict', or 'json'")


def convert_results_v2_to_v3(data):
    """
    Function to convert xarray Dataset results in verion 2 structure to version 3 structure.
    """
    geo1 = Point(float(data['lon']), float(data['lat'])).wkb_hex
    data2 = data.assign_coords({'geometry': geo1}).drop('virtual_station')
    # data2['station_id'].attrs = data['station_id'].attrs
    data2['geometry'].attrs = {'long_name': 'The hexadecimal encoding of the Well-Known Binary (WKB) geometry', 'crs_EPSG': 4326}
    data2.attrs.update({'version': 3})

    data2 = data2.expand_dims('geometry')

    # vars2 = list(data2.variables)

    # if 'name' in vars2:
    #     data2 = data2.assign({'name': (('station_id'), data2['name'])})
    #     data2['name'].attrs = data['name'].attrs
    # if 'ref' in vars2:
    #     data2 = data2.assign({'ref': (('station_id'), data2['ref'])})
    #     data2['ref'].attrs = data['ref'].attrs

    return data2
