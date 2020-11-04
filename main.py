"""
Created on 2020-11-05.

@author: Mike K
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
from utils import get_remote_datasets, get_dataset_params
import schedule

pd.options.display.max_columns = 10


##############################################
### Parameters

base_dir = os.path.realpath(os.path.dirname(__file__))

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)


##############################################
### Class


class tethys(object):
    """

    """
    ## Initial import and assignment function
    def __init__(self, remote_list):
        """

        """









































































