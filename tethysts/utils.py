"""


"""
import sys
import os
import io
import numpy as np
import xarray as xr
import pandas as pd
import orjson
from time import sleep
from datetime import datetime
import copy
import boto3
import botocore
from multiprocessing.pool import ThreadPool
import shapely
from tethys_utils import read_pkl_zstd, list_parse_s3, get_last_date, key_patterns, s3_connection, write_pkl_zstd, read_json_zstd

pd.options.display.max_columns = 10

##############################################
### Helper functions


def get_results_obj_s3(obj_key, connection_config, bucket, max_connections, return_xr=True):
    """

    """
    s3 = s3_connection(connection_config, max_pool_connections=max_connections)

    ts_resp = s3.get_object(Key=obj_key, Bucket=bucket)
    ts_obj = ts_resp.pop('Body')

    if return_xr:
        ts_xr = xr.open_dataset(read_pkl_zstd(ts_obj.read(), False))

        return ts_xr
    else:
        return ts_obj.read()


def result_filters(ts_xr, from_date=None, to_date=None, from_mod_date=None, to_mod_date=None, remove_height=False):
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

    if remove_height:
        ts_xr1 = ts_xr1.squeeze('height').drop('height')

    return ts_xr1


def process_results_output(ts_xr, parameter, modified_date=False, quality_code=False, output='DataArray'):
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
    if output == 'Dataset':
        return ts_xr

    elif output == 'DataArray':
        return ts_xr[out_param]

    elif output == 'Dict':
        darr = ts_xr[out_param]
        data_dict = darr.to_dict()
        if 'name' in data_dict:
            data_dict.pop('name')

        return data_dict

    elif output == 'json':
        darr = ts_xr[out_param]
        data_dict = darr.to_dict()
        if 'name' in data_dict:
            data_dict.pop('name')
        json1 = orjson.dumps(data_dict)

        return json1
    else:
        raise ValueError("output must be one of 'Dataset', 'DataArray', 'Dict', or 'json'")












