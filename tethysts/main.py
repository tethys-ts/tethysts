"""
Created on 2020-11-05.

@author: Mike K
"""
import sys
import os
import io
import numpy as np
import xarray as xr
import pandas as pd
import orjson
# import yaml
from time import sleep
from datetime import datetime
import copy
import boto3
import botocore
from multiprocessing.pool import ThreadPool
import concurrent.futures
import shapely
from tethys_utils import read_pkl_zstd, list_parse_s3, get_last_date, key_patterns, s3_connection, write_pkl_zstd

pd.options.display.max_columns = 10


##############################################
### Parameters

# base_dir = os.path.split(os.path.realpath(os.path.dirname(__file__)))[0]
#
# with open(os.path.join(base_dir, 'parameters.yml')) as param:
#     param = yaml.safe_load(param)
#
# remotes_list = param['remotes']

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
        setattr(self, '_datasets', {})
        setattr(self, '_remotes', {})
        setattr(self, '_stations', {})
        setattr(self, '_key_patterns', key_patterns)

        if isinstance(remotes_list, list):
            datasets = self.get_remotes(remotes_list)

        else:
            pass


    def get_remotes(self, remotes_list, threads=20):
        """

        """
        output = ThreadPool(threads).map(self.get_datasets, remotes_list)

        return self.datasets


    def get_datasets(self, remote):
        """

        """
        s3 = s3_connection(remote['connection_config'])

        try:
            ds_resp = s3.get_object(Key=self._key_patterns['dataset'], Bucket=remote['bucket'])

            ds_obj = ds_resp.pop('Body')
            ds_list = orjson.loads(ds_obj.read())

            ds_list2 = copy.deepcopy(ds_list)
            [l.pop('properties') for l in ds_list2]
            self.datasets.extend(ds_list2)

            ds_dict = {d['dataset_id']: d for d in ds_list}
            remote_dict = {d: {'dataset_id': d, 'bucket': remote['bucket'], 'connection_config': remote['connection_config']} for d in ds_dict}

            self._datasets.update(ds_dict)
            self._remotes.update(remote_dict)

        except:
            print('No datasets.json file in S3 bucket')


    def get_stations(self, dataset_id):
        """

        """
        # dataset = self._datasets_sites[dataset_id]
        # if not hasattr(self, '_datasets_sites'):

        remote = self._remotes[dataset_id]

        s3 = s3_connection(remote['connection_config'])

        site_key = self._key_patterns['station'].format(dataset_id=dataset_id)

        try:
            stn_resp = s3.get_object(Key=site_key, Bucket=remote['bucket'])

            stn_obj = stn_resp.pop('Body')
            stn_list = orjson.loads(read_pkl_zstd(stn_obj.read(), False))

            self._stations.update({dataset_id: {s['station_id']: s for s in stn_list}})

            ## Create spatial index here

            return stn_list

        except:
            print('No stations.json.zst file in S3 bucket')


    def get_time_series_results(self, dataset_id, station_id, from_date=None, to_date=None, from_mod_date=None, to_mod_date=None, modified_date=False, quality_codes=False, output='DataArray'):
        """
        Function to query the time series data given a specific dataset_id and station_id. Multiple optional outputs.

        Parameters
        ----------
        dataset_id : str
            The hashed str of the dataset_id.
        station_id : str
            The hashed string of the station_id.
        from_date : str, Timestamp, datetime, or None
            The start date of the selection.
        to_date : str, Timestamp, datetime, or None
            The end date of the selection.
        from_mod_date : str, Timestamp, datetime, or None
            Only return data post the defined modified date.
        to_mod_date : str, Timestamp, datetime, or None
            Only return data prior to the defined modified date.
        modified_date : bool
            Should the modified dates be returned if they exist?
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
        dataset_stn = self._stations[dataset_id][station_id]
        parameter = self._datasets[dataset_id]['parameter']
        remote = self._remotes[dataset_id]

        obj_info = dataset_stn['time_series_object_key']
        ts_key = obj_info['key']
        bucket = obj_info['bucket']

        s3 = s3_connection(remote['connection_config'])

        try:
            ts_resp = s3.get_object(Key=ts_key, Bucket=bucket)
            ts_obj = ts_resp.pop('Body')
            ts_xr = xr.open_dataset(read_pkl_zstd(ts_obj.read(), False))

            ## Filters
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

            ## Output
            out_param = [parameter]

            if quality_codes:
                if 'quality_codes' in ts_xr1:
                    out_param.extend(['quality_codes'])

            if modified_date:
                if 'modified_date' in ts_xr1:
                    out_param.extend(['modified_date'])

            if len(out_param) == 1:
                out_param = out_param[0]

            ## Return
            if output == 'Dataset':
                return ts_xr1.copy()

            elif output == 'DataArray':
                return ts_xr1[out_param].copy()

            elif output == 'Dict':
                darr = ts_xr1[out_param]
                data_dict = darr.to_dict()
                data_dict.pop('name')

                return data_dict

            elif output == 'json':
                darr = ts_xr1[out_param]
                data_dict = darr.to_dict()
                data_dict.pop('name')
                json1 = orjson.dumps(data_dict)

                return json1
            else:
                raise ValueError("output must be one of 'Dataset', 'DataArray', 'Dict', or 'json'")

        except:
            print('No time series data for dataset_id/station_id combo')


    def bulk_time_series_results(self, dataset_id, station_ids, from_date=None, to_date=None, from_mod_date=None, to_mod_date=None, modified_date=False, quality_codes=False, output='DataArray', threads=10):
        """
        Function to bulk query the time series data given a specific dataset_id and a list of site_ids. Multiple optional outputs.

        Parameters
        ----------
        dataset_id : str
            The hashed str of the dataset_id.
        site_ids : list of str
            A list of hashed strings of the site_ids.
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
        A dictionary of station_id key to a value of whatever the output was set to.
        """
        lister = [(dataset_id, s, from_date, to_date, from_mod_date, to_mod_date, modified_date, quality_codes, output) for s in station_ids]

        output = ThreadPool(threads).starmap(self.get_time_series_results, lister)

        output2 = dict(zip(station_ids, output))

        return output2








######################################
### Testing

# remote = remotes_list[0]
#
# dataset_id = '25e95034d695ac1f9bbfd7d6'
# station_id = '440c5ec714d1c338db0c667b'
# station_ids = ['440c5ec714d1c338db0c667b', 'effdd553968bf602aa9e166d']
#
# dataset_id = '9e1a03dc379cbf7037b0873d'
# site_id = '5c3848a5b9acee6694714e7e'
#
# self = Tethys()
# self = Tethys(remotes_list)
#
# stn_list1 = self.get_stations(dataset_id)
#
# data1 = self.get_time_series_results(dataset_id, station_id, output='Dataset')
# data1 = self.get_time_series_results(dataset_id, station_id, modified_date=True, quality_codes=True, output='DataArray')
# data1 = self.get_time_series_results(dataset_id, station_id, output='Dict')
# data1 = self.get_time_series_results(dataset_id, station_id, from_date='2012-01-02 00:00', output='Dataset')

# data2 = self.bulk_time_series_results(dataset_id, station_ids, output='DataArray')
