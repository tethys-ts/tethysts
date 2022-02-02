"""
Created on 2020-11-05.

@author: Mike K
"""
import os
import requests
import numpy as np
import xarray as xr
import pandas as pd
import orjson
# import yaml
from datetime import datetime
import copy
# from multiprocessing.pool import ThreadPool
import concurrent.futures
import multiprocessing as mp
from tethysts.utils import get_object_s3, result_filters, process_results_output, read_json_zstd, get_nearest_station, get_intersected_stations, spatial_query, convert_results_v2_to_v3, get_nearest_from_extent, read_pkl_zstd, public_remote_key, convert_results_v3_to_v4, s3_client, chunk_filters
# from utils import get_object_s3, result_filters, process_results_output, read_json_zstd, key_patterns, get_nearest_station, get_intersected_stations, spatial_query, convert_results_v2_to_v3, get_nearest_from_extent, read_pkl_zstd, public_remote_key, convert_results_v3_to_v4
from typing import Optional, List, Any, Union
from enum import Enum
import tethys_data_models as tdm
import botocore
from pydantic import HttpUrl

pd.options.display.max_columns = 10


##############################################
### data models



##############################################
### Class


class Tethys(object):
    """
    The base Tethys object.

    Parameters
    ----------
    remotes : list of dict or None
        list of dict of the S3 remotes to access or None which will parse all public datasets.
        The dicts must contain:
        bucket and connection_config.

        bucket : str
            A string of the bucket name.
        connection_config : dict or str
            A dict of strings of service_name, s3, endpoint_url, aws_access_key_id, and aws_secret_access_key. Or it could be a string of the public_url endpoint.

    Returns
    -------
    tethys object
    """
    ## Initial import and assignment function
    def __init__(self, remotes=None):
        """

        """
        setattr(self, 'datasets', [])
        setattr(self, '_datasets', {})
        setattr(self, '_remotes', {})
        setattr(self, '_stations', {})
        setattr(self, '_key_patterns', tdm.utils.key_patterns)
        setattr(self, '_results', {})
        setattr(self, '_results_versions', {})
        setattr(self, '_results_chunks', {})

        if isinstance(remotes, list):
            _ = self.get_datasets(remotes)

        elif remotes is None:
            resp = requests.get(public_remote_key)
            resp.raise_for_status()

            remotes = read_json_zstd(resp.content)
            _ = self.get_datasets(remotes)

        elif remotes != 'pass':
            raise ValueError('remotes must be a list of dict or None.')

        pass


    def get_datasets(self, remotes: List[dict], threads: int = 30):
        """
        The function to get datasets from many remotes.

        Parameters
        ----------
        remotes : list of dict
            list of dict of the S3 remotes to access. The dicts must contain:
            bucket and connection_config.
            bucket : str
                A string of the bucket name.
            connection_config : dict or str
                A dict of strings of service_name, s3, endpoint_url, aws_access_key_id, and aws_secret_access_key. Or it could be a string of the public_url endpoint.
        threads : int
            The number of threads to use. I.e. the number of simultaneous remote reads.

        Returns
        -------
        dict
            of datasets
        """
        ## Validate remotes
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for remote in remotes:
                _ = tdm.base.Remote(**remote)
                f = executor.submit(self.get_remote_datasets, remote)
                futures.append(f)
            _ = concurrent.futures.wait(futures)

        setattr(self, 'remotes', remotes)

        return self.datasets


    def get_remote_datasets(self, remote: dict):
        """
        Get datasets from an individual remote. Saves result into the object.

        Parameters
        ----------
        remote : dict
            dict of the S3 remote to access. The dict must contain:
            bucket and connection_config.
            bucket : str
                A string of the bucket name.
            connection_config : dict or str
                A dict of strings of service_name, s3, endpoint_url, aws_access_key_id, and aws_secret_access_key. Or it could be a string of the public_url endpoint.
            version: int
                The S3 object structure version.

        Returns
        -------
        None
        """
        try:
            get_dict = copy.deepcopy(remote)

            if 'version' in get_dict:
                version = get_dict.pop('version')
            else:
                version = 2

            get_dict['obj_key'] = self._key_patterns[version]['datasets']
            ds_obj = get_object_s3(**get_dict)
            ds_list = read_json_zstd(ds_obj)

            # [l.pop('properties') for l in ds_list2]
            self.datasets.extend(ds_list)

            ds_dict = {d['dataset_id']: d for d in ds_list}
            remote_dict = {d: remote for d in ds_dict}

            self._datasets.update(ds_dict)
            self._remotes.update(remote_dict)

        except:
            print('No datasets.json.zst file in S3 bucket')


    def get_stations(self,
                     dataset_id: str,
                     geometry: Optional[dict] = None,
                     lat: Optional[float] = None,
                     lon: Optional[float] = None,
                     distance: Optional[float] = None,
                     results_object_keys: Optional[bool] = False):
        """
        Method to return the stations associated with a dataset.

        Parameters
        ----------
        dataset_id : str
            The dataset_id of the dataset.
        geometry : dict or None
            A geometry in GeoJSON format. Can be either a point or a polygon. If it's a point, then the method will perform a nearest neighbor query and return one station. If it's a polygon, then the method performs an intersection of all stations within the polygon.
        lat : float or None
            Instead of using the geometry parameter, optionally use lat and lon for the spatial queries. Both lat and lon must be passed for the spatial queries and will override the geometry parameter. If only lat and lon are passed, then the method performs a nearest neighbor query. If distance is passed in addition to lat and lon, then distance is used as a radius buffer and an intersection is performed.
        lon : float or None
            See lat description.
        distance : float or None
            See lat description. This should be in decimal degrees not meters.
        results_object_keys : bool
            Shoud the results object keys be returned? The results object keys list the available results in Tethys.

        Returns
        -------
        list of dict
            of station data
        """
        remote = self._remotes[dataset_id]

        site_key = self._key_patterns[remote['version']]['stations'].format(dataset_id=dataset_id)

        if dataset_id in self._stations:
            stn_dict = copy.deepcopy(self._stations[dataset_id])
        else:
            try:
                stn_obj = get_object_s3(site_key, remote['connection_config'], remote['bucket'])
                stn_list = read_json_zstd(stn_obj)
                stn_dict = {s['station_id']: s for s in stn_list if isinstance(s, dict)}
            except:
                print('No stations.json.zst file in S3 bucket')
                return None

            ## Results obj keys if old version
            if not 'version' in self._datasets[dataset_id]:

                res_obj_keys = {si: s['results_object_key'] for si, s in stn_dict.items()}
                self._results_obj_keys.update({dataset_id: copy.deepcopy(res_obj_keys)})
                [s.update({'results_object_key': s['results_object_key'][-1]}) for i, s in stn_dict.items()]

            self._stations.update({dataset_id: copy.deepcopy(stn_dict)})

        ## Spatial query
        stn_ids = spatial_query(stn_dict, geometry, lat, lon, distance)

        if isinstance(stn_ids, list):
            stn_list1 = [stn_dict[s] for s in stn_ids]
        else:
            stn_list1 = list(stn_dict.values())

        if not results_object_keys:
            [s.pop('results_object_key') for s in stn_list1]

        return stn_list1


    def _get_chunks_versions(self, dataset_id: str):
        """

        """
        if 'system_version' in self._datasets[dataset_id]:
            remote = copy.deepcopy(self._remotes[dataset_id])
            version = remote.pop('version')

            rv_key = self._key_patterns[version]['results_versions'].format(dataset_id=dataset_id)
            remote['obj_key'] = rv_key

            rv_obj = get_object_s3(**remote)
            rv_list = read_json_zstd(rv_obj)

            results_versions = rv_list['results_versions']

            results_chunks = {}
            for s in rv_list['results_chunks']:
                stn_id = s['station_id']
                s['version_date'] = pd.Timestamp(s['version_date']).tz_localize(None)
                if stn_id in results_chunks:
                    results_chunks[stn_id].append(s)
                else:
                    results_chunks[stn_id] = [s]

            self._results_versions[dataset_id] = results_versions
            self._results_chunks[dataset_id] = results_chunks
        else:
            raise NotImplementedError('I need to update this for the previous versions.')
            # _ = self.get_stations(dataset_id)
            # obj_keys = self._results_obj_keys[dataset_id]

        return results_versions, results_chunks


    def get_results_versions(self, dataset_id: str, station_id: str):
        """
        Function to get the run dates of a particular dataset and station.

        Parameters
        ----------
        dataset_id : str
            The dataset_id of the dataset.
        station_id : str
            The station_id of the associated station.

        Returns
        -------
        list
        """
        if dataset_id not in self._results_versions:
            results_versions, results_chunks = self._get_chunks_versions(dataset_id)
        else:
            results_versions = self._results_versions[dataset_id]

        return results_versions


    def _get_results_chunks(self, dataset_id: str, station_id: str, time_interval: int, version_date: Union[str, pd.Timestamp] = None, from_date=None, to_date=None, heights=None, bands=None):
        """

        """
        if dataset_id not in self._results_chunks:
            results_versions, results_chunks = self._get_chunks_versions(dataset_id)
            chunks1 = chunk_filters(results_chunks, time_interval, version_date, from_date, to_date, heights, bands)
        else:
            chunks1 = chunk_filters(self._results_chunks[dataset_id], time_interval, version_date, from_date, to_date, heights, bands)

        return chunks1


    def get_results(self,
                    dataset_id: str,
                    station_ids: Union[str, List[str]] = None,
                    geometry: dict = None,
                    lat: float = None,
                    lon: float = None,
                    from_date: Union[str, pd.Timestamp, datetime] = None,
                    to_date: Union[str, pd.Timestamp, datetime] = None,
                    from_mod_date: Union[str, pd.Timestamp, datetime] = None,
                    to_mod_date: Union[str, pd.Timestamp, datetime] = None,
                    # modified_date: Union[str, pd.Timestamp, datetime, None] = None,
                    # quality_code: Optional[bool] = False,
                    version_date: Union[str, pd.Timestamp, datetime] = None,
                    heights: Union[int, float] = None,
                    bands: int = None,
                    squeeze_dims: bool = False,
                    output: str = 'Dataset',
                    cache: str = None,
                    threads: int = 20):
        """
        Function to query the time series data given a specific dataset_id and station_id. Multiple optional outputs.

        Parameters
        ----------
        dataset_id : str
            The dataset_id of the dataset.
        station_id : str or None
            The station_id of the associated station.
        geometry : dict or None
            A geometry in GeoJSON format. Can be either a point or a polygon. If it's a point, then the method will perform a nearest neighbor query and return one station.
        lat : float or None
            Instead of using the geometry parameter, optionally use lat and lon for the spatial queries. Both lat and lon must be passed for the spatial queries and will override the geometry parameter. If only lat and lon are passed, then the method performs a nearest neighbor query.
        lon : float or None
            See lat description.
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
        quality_code : bool
            Should the quality codes be returned if they exist?
        run_date : str or Timestamp
            The run_date of the results to be returned. Defaults to None which will return the last run date.
        squeeze_dims : bool
            Should all dimensions with a length of one be removed from the parameter's dimensions?
        output : str
            Output format of the results. Options are:
                Dataset - return the entire contents of the netcdf file as an xarray Dataset,
                DataArray - return the requested dataset parameter as an xarray DataArray,
                Dict - return a dictionary of results from the DataArray,
                json - return a json str of the Dict.
        cache : str or None
            How the results should be cached. Current options are None (which does not cache) and 'memory' (which caches the results in the Tethys object in memory).

        Returns
        -------
        Whatever the output was set to.
        """

        ## Get parameters
        dataset = self._datasets[dataset_id]
        parameter = dataset['parameter']
        remote = copy.deepcopy(self._remotes[dataset_id])
        version = remote.pop('version')
        time_interval = int(dataset['chunk_parameters']['time_interval'])

        if isinstance(geometry, dict):
            geom_type = geometry['type']
        else:
            geom_type = None

        if isinstance(station_ids, str):
            stn_ids = [station_ids]
        elif isinstance(station_ids, list):
            stn_ids = station_ids
        elif ((geom_type in ['Point', 'Polygon']) or (isinstance(lat, float) and isinstance(lon, float))):
            ## Get all stations
            if dataset_id not in self._stations:
                _ = self.get_stations(dataset_id)

            stn_dict = self._stations[dataset_id]

            # Run the spatial query
            stn_ids = spatial_query(stn_dict, geometry, lat, lon)
        else:
            raise ValueError('A station_id, geometry or a combination of lat and lon must be passed.')

        ## Get results chunks
        chunk_keys = []
        extend = chunk_keys.extend
        for stn_id in stn_ids:
            c1 = self._get_results_chunks(dataset_id, stn_id, time_interval, version_date, from_date, to_date, heights, bands)
            extend([c['key'] for c in c1])


        # with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        #     futures = []
        #     for stn_id in stn_ids:
        #         f = executor.submit(self._get_results_chunks, dataset_id, stn_id, time_interval, version_date, from_date, to_date, heights, bands)
        #         futures.append(f)
        #     runs = concurrent.futures.wait(futures)

        # chunk_keys1 = [r.result() for r in runs[0]]
        # chunk_keys = []
        # extend = chunk_keys.extend
        # for chunk in chunk_keys1:
        #     extend([c['key'] for c in chunk])

        chunk_keys.sort()

        ## Get results chunks
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for key in chunk_keys:
                remote['obj_key'] = key
                f = executor.submit(get_object_s3, **remote)
                futures.append(f)
            runs = concurrent.futures.wait(futures)

        chunks1 = [r.result() for r in runs[0]]

        # TODO: still need to finish this and add caching

        # cache results if requested
        # if cache == 'memory':
        #     if obj_key in self._results:
        #         new_len = len(ts_obj)
        #         old_len = len(self._results[obj_key])
        #         if new_len != old_len:
        #             self._results[obj_key] = ts_obj
        #     else:
        #         self._results[obj_key] = ts_obj

        # Open results
        xr3 = xr.open_dataset(read_pkl_zstd(ts_obj))

        ## Convert to new version
        attrs = xr3.attrs.copy()
        if ('version' not in attrs):
            xr3 = convert_results_v2_to_v3(xr3)
            attrs['version'] = 3

        if attrs['version'] == 3:
            xr3 = convert_results_v3_to_v4(xr3)

        ## Extra spatial query if data are stored in blocks
        if ('station_geometry' in xr3) and ((geom_type == 'Point') or (isinstance(lat, float) and isinstance(lon, float))):
            xr3 = get_nearest_from_extent(xr3, geometry, lat, lon)

        ## Filters
        ts_xr1 = result_filters(xr3, from_date, to_date, from_mod_date, to_mod_date)

        # if not 'station_id' in list(ts_xr1.coords):
        #     ts_xr1 = ts_xr1.expand_dims('station_id').set_coords('station_id')

        ## Output
        output1 = process_results_output(ts_xr1, parameter, modified_date, quality_code, output, squeeze_dims)

        return output1


    def get_bulk_results(self,
                         dataset_id: str,
                         station_ids: List[str],
                         geometry: Optional[dict] = None,
                         lat: Optional[float] = None,
                         lon: Optional[float] = None,
                         from_date: Union[str, pd.Timestamp, datetime, None] = None,
                         to_date: Union[str, pd.Timestamp, datetime, None] = None,
                         from_mod_date: Union[str, pd.Timestamp, datetime, None] = None,
                         to_mod_date: Union[str, pd.Timestamp, datetime, None] = None,
                         modified_date: Optional[bool] = False,
                         quality_code: Optional[bool] = False,
                         run_date: Union[str, pd.Timestamp, datetime, None] = None,
                         squeeze_dims: Optional[bool] = False,
                         output: str = 'Dataset',
                         cache: Optional[str] = None,
                         threads: int = 30):
        """
        Function to bulk query the time series data given a specific dataset_id and a list of station_ids. The output will be specified by the output parameter and will be concatenated along the station_id dimension.

        Parameters
        ----------
        dataset_id : str
            The hashed str of the dataset_id.
        station_ids : list of str
            A list of hashed str of the site_ids.
        geometry : dict or None
            A geometry in GeoJSON format. Can be either a point or a polygon. If it's a point, then the method will perform a nearest neighbor query and return one station.
        lat : float or None
            Instead of using the geometry parameter, optionally use lat and lon for the spatial queries. Both lat and lon must be passed for the spatial queries and will override the geometry parameter. If only lat and lon are passed, then the method performs a nearest neighbor query.
        lon : float or None
            See lat description.
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
        quality_code : bool
            Should the quality codes be returned if they exist?
        run_date : str or Timestamp
            The run_date of the results to be returned. Defaults to None which will return the last run date.
        squeeze_dims : bool
            Should all dimensions with a length of one be removed from the parameter's dimensions?
        output : str
            Output format of the results. Options are:
                Dataset - return the entire contents of the netcdf file as an xarray Dataset,
                DataArray - return the requested dataset parameter as an xarray DataArray,
                Dict - return a dictionary of results from the DataArray,
                json - return a json str of the Dict.
        cache : str or None
            How the results should be cached. Current options are None (which does not cache) and 'memory' (which caches the results in the Tethys object in memory).
        threads : int
            The number of simultaneous downloads.

        Returns
        -------
        Format specified by the output parameter
            Will be concatenated along the station_id dimension
        """
        dataset = self._datasets[dataset_id]
        parameter = dataset['parameter']

        lister = [(dataset_id, s, geometry, lat, lon, from_date, to_date, from_mod_date, to_mod_date, modified_date, quality_code, run_date, False, 'Dataset', cache) for s in station_ids]

        output1 = ThreadPool(threads).starmap(self.get_results, lister)
        # output2 = [d if 'station_id' in list(d.coords) else d.expand_dims('station_id').set_coords('station_id') for d in output1]

        if 'geometry' in output1[0]:
            # deal with the situation where the variable names are not the same for all datasets
            try:
                xr_ds1 = xr.combine_nested(output1, 'geometry', data_vars='minimal', combine_attrs="override")
            except:
                xr_ds1 = xr.merge(output1, combine_attrs="override")
        else:
            try:
                xr_ds1 = xr.combine_by_coords(output1, data_vars='minimal')
            except:
                xr_ds1 = xr.merge(output1, combine_attrs="override")

        ## Output
        output3 = process_results_output(xr_ds1, parameter, modified_date, quality_code, output, squeeze_dims)

        return output3



######################################
### Testing
