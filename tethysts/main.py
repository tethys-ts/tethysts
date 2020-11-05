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
# from pymongo import MongoClient, InsertOne, DeleteOne, ReplaceOne, UpdateOne
# from pymongo.errors import BulkWriteError
import pandas as pd
import orjson
from time import sleep
from datetime import datetime
import copy
import boto3
import botocore
from multiprocessing.pool import ThreadPool
import concurrent.futures
import shapely
# from bson.objectid import ObjectId
from tethys_utils import read_pkl_zstd, list_parse_s3, get_last_date, key_patterns, s3_connection, write_pkl_zstd
# from utils import get_remote_datasets, get_dataset_params

pd.options.display.max_columns = 10


##############################################
### Parameters

# import yaml
# base_dir = os.path.realpath(os.path.dirname(__file__))
#
# with open(os.path.join(base_dir, 'parameters.yml')) as param:
#     param = yaml.safe_load(param)
#
#
# remotes_list = param['remotes']

dataset_key = key_patterns['dataset']

##############################################
### Class


class Tethys(object):
    """

    """
    ## Initial import and assignment function
    def __init__(self, remotes_list=None):
        """

        """
        setattr(self, 'datasets', [])
        setattr(self, '_datasets_sites', {})
        setattr(self, '_dataset_key', dataset_key)

        if isinstance(remotes_list, list):
            datasets = self.get_remotes_list(remotes_list)

        else:
            pass


    def get_remotes_list(self, remotes_list, threads=20):
        """

        """
        output = ThreadPool(threads).map(self.get_dataset_list, remotes_list)

        return self.datasets


    def get_dataset_list(self, remote):
        """

        """
        s3 = s3_connection(remote['connection_config'])

        try:
            ds_resp = s3.get_object(Key=self._dataset_key, Bucket=remote['bucket'])

            ds_obj = ds_resp.pop('Body')
            ds_list = orjson.loads(ds_obj.read())

            ds_list2 = copy.deepcopy(ds_list)
            [l.pop('properties') for l in ds_list2]
            self.datasets.append(ds_list2)

            ds_dict = {d['dataset_id']: {'dataset': d, 'remote': remote} for d in ds_list}

            self._datasets_sites.update(ds_dict)

        except:
            print('No datasets.json file in S3 bucket')


    def get_sites_list(self, dataset_id):
        """

        """
        # dataset = self._datasets_sites[dataset_id]
        # if not hasattr(self, '_datasets_sites'):

        remote = self._datasets_sites[dataset_id]['remote']

        s3 = s3_connection(remote['connection_config'])

        site_key = key_patterns['site'].format(dataset_id=dataset_id)

        try:
            site_resp = s3.get_object(Key=site_key, Bucket=remote['bucket'])

            site_obj = site_resp.pop('Body')
            site_list = orjson.loads(read_pkl_zstd(site_obj.read(), False))

            self._datasets_sites[dataset_id].update({'sites': {s['site_id']: s for s in site_list}})

            ## Create spatial index here

            return site_list

        except:
            print('No sites.json.zst file in S3 bucket')


    def get_time_series_results(self, dataset_id, site_id, from_date=None, to_date=None, quality_codes=False, output='DataArray'):
        """
        Function to query the time series data given a specific dataset_id and site_id. Multiple optional outputs.

        Parameters
        ----------
        dataset_id : str
            The hashed str of the dataset_id.
        site_id : str
            The hashed string of the site_id.
        from_date : str, Timestamp, datetime, or None
            The start date of the selection.
        to_date : str, Timestamp, datetime, or None
            The end date of the selection.
        quality_codes : bool
            Should the quality codes be returned if they exist?
        output : str
            Output format of the results. Options are:
            Dataset - return the entire contents of the netcdf file as an xarray Dataset,
            DataArray - return the requested dataset parameter as an xarray DataArray,
            Dict - return a dictionary of results from the DataArray,
            json - return a json str of the Dict.

        Returns
        -------
        Whatever the output was set to.
        """
        dataset_dict = self._datasets_sites[dataset_id]
        dataset_site = dataset_dict['sites'][site_id]
        dataset = dataset_dict['dataset']
        parameter = dataset['parameter']

        obj_info = dataset_site['time_series_object_info']
        ts_key = obj_info['key']
        bucket = obj_info['bucket']

        s3 = s3_connection(dataset_dict['remote']['connection_config'])

        try:
            ts_resp = s3.get_object(Key=ts_key, Bucket=bucket)
            ts_obj = ts_resp.pop('Body')
            ts_xr = xr.open_dataset(read_pkl_zstd(ts_obj.read(), False))

            if isinstance(from_date, (str, pd.Timestamp, datetime)):
                from_date1 = pd.Timestamp(from_date)
            else:
                from_date1 = None
            if isinstance(to_date, (str, pd.Timestamp, datetime)):
                to_date1 = pd.Timestamp(to_date)
            else:
                to_date1 = None

            if (to_date1 is not None) or (from_date1 is not None):
                ts_xr1 = ts_xr.sel(time=slice(from_date1, to_date1))
            else:
                ts_xr1 = ts_xr

            ## Return
            if output == 'Dataset':
                return ts_xr1.copy()

            elif output == 'DataArray':
                return ts_xr1[parameter].copy()

            elif output == 'Dict':
                darr = ts_xr1[parameter]
                data_dict = darr.to_dict()
                data_dict.pop('name')

                return data_dict

            elif output == 'json':
                darr = ts_xr1[parameter]
                data_dict = darr.to_dict()
                data_dict.pop('name')
                json1 = orjson.dumps(data_dict)

                return json1
            else:
                raise ValueError("output must be one of 'Dataset', 'DataArray', 'Dict', or 'json'")

        except:
            print('No time series data for dataset_id/site_id combo')


    # def bulk_time_series_results(self, dataset_id, site_ids, from_date=None, to_date=None, quality_codes=False, output='DataArray'):
    #     """
    #     Function to bulk query the time series data given a specific dataset_id and a list of site_ids. Multiple optional outputs.
    #
    #     Parameters
    #     ----------
    #     dataset_id : str
    #         The hashed str of the dataset_id.
    #     site_ids : list of str
    #         A list of hashed strings of the site_ids.
    #     from_date : str, Timestamp, datetime, or None
    #         The start date of the selection.
    #     to_date : str, Timestamp, datetime, or None
    #         The end date of the selection.
    #     quality_codes : bool
    #         Should the quality codes be returned if they exist?
    #     output : str
    #         Output format of the results. Options are:
    #         Dataset - return the entire contents of the netcdf file as an xarray Dataset,
    #         DataArray - return the requested dataset parameter as an xarray DataArray,
    #         Dict - return a dictionary of results from the DataArray,
    #         json - return a json str of the Dict.
    #
    #     Returns
    #     -------
    #     Whatever the output was set to.
    #     """
    #     lister = [(dataset_id, s, from_date, to_date, quality_codes, 'Dataset') for s in site_ids]
    #
    #     output = ThreadPool(4).starmap(self.get_time_series_results, lister)
    #
    #     ds1 = xr.concat(output, dim='time')









######################################
### Testing

# remote = remotes_list[0]
#
# dataset_id = 'cbba7575fb51024f4bf961e2'
# site_id = 'b7c99b99c209c70a946472fd'
# site_ids = ['b7c99b99c209c70a946472fd', '76cf3a75b64396ed21af3cb5']
#
# dataset_id = '9e1a03dc379cbf7037b0873d'
# site_id = '5c3848a5b9acee6694714e7e'
#
# self = Tethys()
# self = Tethys(remotes_list)
#
# site_list1 = self.get_sites_list(dataset_id)
#
# data1 = self.get_time_series_results(dataset_id, site_id, output='Dataset')
# data1 = self.get_time_series_results(dataset_id, site_id, output='Dict')
# data1 = self.get_time_series_results(dataset_id, site_id, from_date='2012-01-02 00:00', output='Dataset')

# data2 = self.bulk_time_series_results(dataset_id, site_ids, output='DataArray')
