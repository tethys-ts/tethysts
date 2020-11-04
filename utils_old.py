# -*- coding: utf-8 -*-
"""
Created on 2020-10-04

@author: MichaelEK
"""
import sys
import os
import io
import numpy as np
# base_dir = os.path.realpath(os.path.dirname(__file__))
# sys.path.append(base_dir)
import xarray as xr
from pymongo import MongoClient, InsertOne, DeleteOne, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError
import pandas as pd
import yaml
from time import sleep
import copy
import boto3
import botocore
from multiprocessing.pool import ThreadPool
from bson.objectid import ObjectId
from tethys_utils import read_pkl_zstd, list_parse_s3, get_last_date, ts_key_pattern, s3_connection

pd.options.display.max_columns = 10


################################################
### Functions


def get_remote_datasets(param):
    """

    """
    remotes = param['remotes']

    datasets_detailed = []

    for l in remotes:
        s3_config = l['connection_config'].copy()
        s3 = s3_connection(s3_config)
        delimiter = l['delimiter']
        skp1 = 'time_series' + delimiter
        df1 = list_parse_s3(s3, l['bucket'], skp1)
        if not df1.empty:
            df_list = df1.Key.apply(lambda x: x.split(delimiter)[1:])

            df2 = pd.DataFrame(item[:8] + [item[-1].split('.')[1]] for item in df_list if '.' in item[-1])
            df3 = df2.drop_duplicates()
            df3.columns = ['owner', 'feature', 'parameter', 'method', 'processing_code', 'aggregation_statistic', 'frequency_interval', 'utc_offset', 'time_series_type']

            list1 = df3.to_dict(orient='records')
            for l1 in list1:
                prop = copy.deepcopy(l)
                prop.update({'time_series_type': l1.pop('time_series_type')})
                l1.update({'properties': prop})

            datasets_detailed.extend(list1)

    return datasets_detailed


def get_dataset_params(base_dir, dataset_file='input.yml'):
    """

    """
    with open(os.path.join(base_dir, dataset_file)) as param:
        param = yaml.safe_load(param)

    if 'datasets' in param:
        requested_datasets = param['datasets']
    else:
        requested_datasets = None

    if 'sites' in param:
        requested_sites = param['sites']
    else:
        requested_sites = None

    if 'scheduling' in param:
        scheduling = param['scheduling']
    else:
        scheduling = {'delay': 10}

    if 'from_mod_date' in param:
        from_mod_date = param['from_mod_date']
    else:
        from_mod_date = None

    if 'remotes' in param:
        extra_remotes = param['remotes']
    else:
        extra_remotes = None

    return requested_datasets, requested_sites, scheduling, from_mod_date, extra_remotes

























































