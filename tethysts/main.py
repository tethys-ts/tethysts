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
from tethysts.utils import get_object_s3, result_filters, process_results_output, read_json_zstd, get_nearest_station, get_intersected_stations, spatial_query, get_nearest_from_extent, read_pkl_zstd, public_remote_key, s3_client, chunk_filters, download_results, make_run_date_key, update_nested, nest_results
# from utils import get_object_s3, result_filters, process_results_output, read_json_zstd, key_patterns, get_nearest_station, get_intersected_stations, spatial_query, convert_results_v2_to_v3, get_nearest_from_extent, read_pkl_zstd, public_remote_key, convert_results_v3_to_v4
from typing import List, Union
import tethys_data_models as tdm
import pathlib
from time import time
# import pymongo
# pymongo.database.Database

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
    def __init__(self, remotes: List[tdm.base.Remote] = None, cache: Union[pathlib.Path] = None):
        """
        The cache parameter might eventually include pymongo.database.Database.

        Parameters
        ----------
        remotes : list of dict
            list of dict of the S3 remotes to access. The dicts must contain:
            bucket and connection_config.
            bucket : str
                A string of the bucket name.
            connection_config : dict or None
                A dict of strings of service_name, s3, endpoint_url, aws_access_key_id, and aws_secret_access_key.
            public_url : str or None
                The base public Http URL to download data from S3.
            version: int
                The system version number.
        cache : str path or None
            If the input is a str path, then data will be cached locally. None will perform no caching.


        """
        setattr(self, 'datasets', [])
        setattr(self, '_datasets', {})
        setattr(self, '_remotes', {})
        setattr(self, '_stations', {})
        setattr(self, '_key_patterns', tdm.utils.key_patterns)
        # setattr(self, '_results', {})
        setattr(self, '_versions', {})
        setattr(self, '_results_chunks', {})

        if isinstance(cache, str):
            cache_path = pathlib.Path(cache)
            os.makedirs(cache_path, exist_ok=True)
            setattr(self, 'cache', cache_path)
        else:
            setattr(self, 'cache', None)

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
            connection_config : dict or None
                A dict of strings of service_name, s3, endpoint_url, aws_access_key_id, and aws_secret_access_key.
            public_url : str or None
                The base public Http URL to download data from S3.
            version: int
                The system version number.
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
                # if not 'version' in remote:
                #     remote['version'] = 2
                remote_m = orjson.loads(tdm.base.Remote(**remote).json(exclude_none=True))
                if 'description' in remote_m:
                    _ = remote_m.pop('description')
                f = executor.submit(self._load_remote_datasets, remote_m)
                futures.append(f)
            _ = concurrent.futures.wait(futures)

        setattr(self, 'remotes', remotes)

        return self.datasets


    def _load_remote_datasets(self, remote: dict):
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

            version = get_dict.pop('version')

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
                     geometry: dict = None,
                     lat: float = None,
                     lon: float = None,
                     distance: float = None,
                     version_date: Union[str, datetime, pd.Timestamp] = None
                     ):
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
        version_date: str in iso 8601 datetime format or None
            The specific version of the stations data. None will return the latest version.

        Returns
        -------
        list of dict
            of station data
        """
        remote = copy.deepcopy(self._remotes[dataset_id])
        version = remote.pop('version')

        vd = self._get_version_date(dataset_id, version_date)
        stn_key = self._get_stns_rc_key(dataset_id, 'stations', vd)

        run_get = True

        if dataset_id in self._stations:
            if vd in self._stations[dataset_id]:
                stn_dict = copy.deepcopy(self._stations[dataset_id][vd])
                run_get = False

        if run_get:
            try:
                remote['obj_key'] = stn_key
                stn_obj = get_object_s3(**remote)
                stn_list = read_json_zstd(stn_obj)

                stn_dict = {s['station_id']: s for s in stn_list if isinstance(s, dict)}
                update_nested(self._stations, dataset_id, vd, stn_dict)
            except:
                print('No stations.json.zst file in S3 bucket')
                return None

        ## Spatial query
        stn_ids = spatial_query(stn_dict, geometry, lat, lon, distance)

        if isinstance(stn_ids, list):
            stn_list1 = [stn_dict[s] for s in stn_ids]
        else:
            stn_list1 = list(stn_dict.values())

        return stn_list1


    def _get_stns_rc_key(self, dataset_id: str, key_name, version_date: str = None):
        """

        """
        if key_name not in ['results_chunks', 'stations']:
            raise ValueError('key_name must be either results_chunks or stations.')

        remote = copy.deepcopy(self._remotes[dataset_id])
        system_version = remote.pop('version')

        # Check version_date
        version_date = self._get_version_date(dataset_id, version_date)

        vd_key = make_run_date_key(version_date)
        stn_key = self._key_patterns[system_version][key_name].format(dataset_id=dataset_id, version_date=vd_key)

        return stn_key


    def _get_results_chunks(self, dataset_id: str, version_date: str = None):
        """

        """
        remote = copy.deepcopy(self._remotes[dataset_id])
        system_version = remote.pop('version')

        run_get = True

        if dataset_id in self._results_chunks:
            if version_date in self._results_chunks[dataset_id]:
                rc_list = self._results_chunks[dataset_id][version_date]
                run_get = False

        if run_get:
            rc_key = self._get_stns_rc_key(dataset_id, 'results_chunks', version_date)

            remote1 = copy.deepcopy(remote)

            remote1['obj_key'] = rc_key
            stn_obj = get_object_s3(**remote1)

            rc_list = read_json_zstd(stn_obj)

            update_nested(self._results_chunks, dataset_id, version_date, rc_list)

        return rc_list


    # def _get_v2_v3_chunks_versions(self, dataset_id: str, remote, system_version):
    #     """

    #     """
    #     rok_key = self._key_patterns[system_version]['results_object_keys'].format(dataset_id=dataset_id)
    #     remote['obj_key'] = rok_key

    #     rok_obj = get_object_s3(**remote)
    #     rok_list = read_json_zstd(rok_obj)

    #     results_versions, results_chunks = v2_v3_results_chunks(rok_list)

    #     vd = results_versions[-1]['version_date']

    #     self._versions[dataset_id] = results_versions
    #     update_nested(self._results_chunks, dataset_id, vd, results_chunks)

    #     return results_chunks


    def get_versions(self, dataset_id: str):
        """
        Function to get the versions of a particular dataset.

        Parameters
        ----------
        dataset_id : str
            The dataset_id of the dataset.

        Returns
        -------
        list
        """
        if dataset_id not in self._versions:
            remote = copy.deepcopy(self._remotes[dataset_id])
            version = remote.pop('version')

            rv_key = self._key_patterns[version]['versions'].format(dataset_id=dataset_id)
            remote['obj_key'] = rv_key

            rv_obj = get_object_s3(**remote)
            rv_list = read_json_zstd(rv_obj)

            self._versions[dataset_id] = rv_list

        versions = self._versions[dataset_id]

        return versions


    def _get_version_date(self, dataset_id: str, version_date: Union[str, pd.Timestamp] = None):
        """

        """
        if dataset_id not in self._versions:
            versions = self.get_versions(dataset_id)
        else:
            versions = self._versions[dataset_id]

        if version_date is None:
            vd = versions[-1]['version_date']
        else:
            vd = pd.Timestamp(version_date).tz_localize(None).isoformat()
            vd_list = [v for v in versions if v['version_date'] == vd]
            if len(vd_list) == 0:
                raise ValueError('version_date is not available.')

        return vd


    # def _get_results_chunks_filter(self, dataset_id: str, station_id: str, time_interval: int, version_date: Union[str, pd.Timestamp], from_date=None, to_date=None, heights=None, bands=None):
    #     """

    #     """
    #     chunks1 = chunk_filters(self._results_chunks[dataset_id][station_id], version_date, time_interval, from_date, to_date, heights, bands)

    #     return chunks1


    def clear_cache(self, max_size=1000, max_age=7):
        """
        Clears the cache based on specified max_size and max_age. The cache path must be assigned at the Tethys initialisation for this function to work.

        Parameters
        ----------
        max_size: int
            The total maximum size of all files in the cache in MBs.
        max_age: int or float
            The maximum age of files in the cache in days.

        Returns
        -------
        None
        """
        if not isinstance(self.cache, pathlib.Path):
            raise TypeError('The cache path must be set when initialising Tethys.')

        cache_gen1 = list(self.cache.rglob('*.nc'))

        stats1 = []
        for c in cache_gen1:
            stats = c.stat()
            stats1.append([str(c), stats.st_size, stats.st_mtime])

        stats2 = pd.DataFrame(stats1, columns=['file_path', 'file_size', 'mtime'])
        stats2['mtime'] = pd.to_datetime(stats2['mtime'], unit='s').round('s')

        stats2 = stats2.sort_values('mtime')
        stats2['file_cumsum'] = stats2['file_size'].cumsum()

        now1 = pd.Timestamp.now().round('s')
        then1 = now1 - pd.DateOffset(days=max_age)

        rem1_bool = stats2['mtime'] < then1
        rem2_bool = stats2['file_cumsum'] > (max_size*1000000)

        rem_files = stats2[rem1_bool | rem2_bool]['file_path'].tolist()

        if rem_files:
            for f in rem_files:
                os.remove(f)


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
                    version_date: Union[str, pd.Timestamp, datetime] = None,
                    heights: Union[List[Union[int, float]], Union[int, float]] = None,
                    bands: Union[List[int], int] = None,
                    squeeze_dims: bool = False,
                    output: str = 'xarray',
                    threads: int = 30,
                    # include_chunk_vars: bool = False
                    ):
        """
        Function to query the results data given a specific dataset_id and station_ids. Multiple optional outputs.

        Parameters
        ----------
        dataset_id : str
            The dataset_id of the dataset.
        station_ids : str, list of str, or None
            The station_ids of the associated station.
        geometry : dict or None
            A point geometry in GeoJSON format. If it's a point, then the method will perform a nearest neighbor query and return one station.
        lat : float or None
            Instead of using the geometry parameter, optionally use lat and lon for the nearest neighbor spatial query. Both lat and lon must be passed for the spatial query and will override the geometry parameter.
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
        version_date : str or Timestamp
            The version date of the results to be returned. Defaults to None which will return the last version.
        heights : list of int/float, int/float, or None
            The heights to return. If None, then all heights are returned.
        bands : list of int, int, or None
            The bands to return. If None, then all bands are returned.
        squeeze_dims : bool
            Should all dimensions with a length of one be removed from the parameter's dimensions?
        output : str
            Output format of the results. Options are:
                xarray - return the entire contents of the netcdf file as an xarray Dataset,
                dict - return a dictionary of results from the DataArray,
                json - return a json str of the dict.

        Returns
        -------
        Whatever the output was set to.
        """
        ## Get parameters
        dataset = self._datasets[dataset_id]
        parameter = dataset['parameter']
        if 'result_type' in dataset:
            result_type = dataset['result_type']
        else:
            result_type = ''
        remote = copy.deepcopy(self._remotes[dataset_id])
        version = remote.pop('version')

        vd = self._get_version_date(dataset_id, version_date)

        if 'chunk_parameters' in dataset:
            time_interval = int(dataset['chunk_parameters']['time_interval'])
        else:
            time_interval = 0

        if isinstance(geometry, dict):
            geom_type = geometry['type']
        else:
            geom_type = None

        if isinstance(station_ids, str):
            stn_ids = [station_ids]
        elif isinstance(station_ids, list):
            stn_ids = station_ids
        elif ((geom_type in ['Point']) or (isinstance(lat, float) and isinstance(lon, float))):
            ## Get all stations
            if dataset_id not in self._stations:
                _ = self.get_stations(dataset_id, version_date=vd)

            stn_dict = self._stations[dataset_id][vd]

            # Run the spatial query
            stn_ids = spatial_query(stn_dict, geometry, lat, lon, None)
        else:
            raise ValueError('A station_id, point geometry or a combination of lat and lon must be passed.')

        ## Get results chunks
        rc_list = self._get_results_chunks(dataset_id, vd)

        chunks = chunk_filters(rc_list, stn_ids, time_interval, from_date, to_date, heights, bands)
        chunks, index, dims = nest_results(chunks)

        ## Get results chunks
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            remote['cache'] = self.cache
            if not 'public_url' in remote:
                s3 = s3_client(remote['connection_config'], threads)
                remote['s3'] = s3

            futures = []
            for chunk in chunks:
                remote['chunk'] = chunk
                remote['index'] = index
                remote['dims'] = dims
                remote['from_date'] = from_date
                remote['to_date'] = to_date
                f = executor.submit(download_results, **remote)
                futures.append(f)
            _ = concurrent.futures.wait(futures)

        ## Open results
        # TODO: definitely not happy with the performance of combining datasets
        # I will probably need to organise and combine the objects one dimension at a time iteratively.

        dim_len = len(dims)
        if dim_len == 1:
            temp1 = index[stn_ids[0]][0]
        elif dim_len == 2:
            temp1 = index[stn_ids[0]][0][0]
        elif dim_len == 3:
            temp1 = index[stn_ids[0]][0][0][0]
        encoding = {v: temp1[v].encoding for v in list(temp1.variables) if ('chunk' not in v)}

        if 'chunk_day' in dims:
            time_pos = dims.index('chunk_day')
            concat_dims = [dim for dim in dims if dim != 'chunk_day']
            concat_dims.insert(time_pos, 'time')

        groups1 = []
        for stn_id, data_list in index.items():
            xr1 = xr.combine_nested(data_list, concat_dims, data_vars='minimal', coords='minimal', combine_attrs='override', compat='override').load()
            groups1.append(xr1)

        del index

        xr3 = groups1[0]

        if len(groups1) > 1:
            for c in groups1[1:]:
                xr3 = xr3.combine_first(c)

        del groups1

        ## Add the encodings back and correct the float that should be int
        for v, enc in encoding.items():
            if v in xr3:
                _ = [enc.pop(d) for d in ['original_shape', 'source'] if d in enc]
                xr3[v].encoding = enc

                dtype = enc['dtype'].name

                if ('int' in dtype) and (not 'scale_factor' in enc) and (not 'calendar' in enc):
                    xr3[v] = xr3[v].astype(dtype)

        ## Convert to new version
        attrs = xr3.attrs.copy()
        if 'version' in attrs:
            attrs['system_version'] = attrs.pop('version')

        ## Extra spatial query if data are stored in blocks
        if ('grid' in result_type) and ((geom_type == 'Point') or (isinstance(lat, float) and isinstance(lon, float))):
            xr3 = get_nearest_from_extent(xr3, geometry, lat, lon)

        ## Filters
        ts_xr1 = result_filters(xr3, from_mod_date=from_mod_date, to_mod_date=to_mod_date)

        ts_xr1.attrs['version_date'] = pd.Timestamp(vd).tz_localize(None).isoformat()

        ## Output
        ts_xr1 = process_results_output(ts_xr1, parameter, modified_date=False, quality_code=False, output=output, squeeze_dims=squeeze_dims)

        return ts_xr1


    # def get_bulk_results(self,
    #                      dataset_id: str,
    #                      station_ids: List[str],
    #                      geometry: Optional[dict] = None,
    #                      lat: Optional[float] = None,
    #                      lon: Optional[float] = None,
    #                      from_date: Union[str, pd.Timestamp, datetime, None] = None,
    #                      to_date: Union[str, pd.Timestamp, datetime, None] = None,
    #                      from_mod_date: Union[str, pd.Timestamp, datetime, None] = None,
    #                      to_mod_date: Union[str, pd.Timestamp, datetime, None] = None,
    #                      modified_date: Optional[bool] = False,
    #                      quality_code: Optional[bool] = False,
    #                      run_date: Union[str, pd.Timestamp, datetime, None] = None,
    #                      squeeze_dims: Optional[bool] = False,
    #                      output: str = 'Dataset',
    #                      cache: Optional[str] = None,
    #                      threads: int = 30):
    #     """
    #     Function to bulk query the time series data given a specific dataset_id and a list of station_ids. The output will be specified by the output parameter and will be concatenated along the station_id dimension.

    #     Parameters
    #     ----------
    #     dataset_id : str
    #         The hashed str of the dataset_id.
    #     station_ids : list of str
    #         A list of hashed str of the site_ids.
    #     geometry : dict or None
    #         A geometry in GeoJSON format. Can be either a point or a polygon. If it's a point, then the method will perform a nearest neighbor query and return one station.
    #     lat : float or None
    #         Instead of using the geometry parameter, optionally use lat and lon for the spatial queries. Both lat and lon must be passed for the spatial queries and will override the geometry parameter. If only lat and lon are passed, then the method performs a nearest neighbor query.
    #     lon : float or None
    #         See lat description.
    #     from_date : str, Timestamp, datetime, or None
    #         The start date of the selection.
    #     to_date : str, Timestamp, datetime, or None
    #         The end date of the selection.
    #     from_mod_date : str, Timestamp, datetime, or None
    #         Only return data post the defined modified date.
    #     to_mod_date : str, Timestamp, datetime, or None
    #         Only return data prior to the defined modified date.
    #     modified_date : bool
    #         Should the modified dates be returned if they exist?
    #     quality_code : bool
    #         Should the quality codes be returned if they exist?
    #     run_date : str or Timestamp
    #         The run_date of the results to be returned. Defaults to None which will return the last run date.
    #     squeeze_dims : bool
    #         Should all dimensions with a length of one be removed from the parameter's dimensions?
    #     output : str
    #         Output format of the results. Options are:
    #             Dataset - return the entire contents of the netcdf file as an xarray Dataset,
    #             DataArray - return the requested dataset parameter as an xarray DataArray,
    #             Dict - return a dictionary of results from the DataArray,
    #             json - return a json str of the Dict.
    #     cache : str or None
    #         How the results should be cached. Current options are None (which does not cache) and 'memory' (which caches the results in the Tethys object in memory).
    #     threads : int
    #         The number of simultaneous downloads.

    #     Returns
    #     -------
    #     Format specified by the output parameter
    #         Will be concatenated along the station_id dimension
    #     """
    #     dataset = self._datasets[dataset_id]
    #     parameter = dataset['parameter']

    #     lister = [(dataset_id, s, geometry, lat, lon, from_date, to_date, from_mod_date, to_mod_date, modified_date, quality_code, run_date, False, 'Dataset', cache) for s in station_ids]

    #     output1 = ThreadPool(threads).starmap(self.get_results, lister)
    #     # output2 = [d if 'station_id' in list(d.coords) else d.expand_dims('station_id').set_coords('station_id') for d in output1]

    #     if 'geometry' in output1[0]:
    #         # deal with the situation where the variable names are not the same for all datasets
    #         try:
    #             xr_ds1 = xr.combine_nested(output1, 'geometry', data_vars='minimal', combine_attrs="override")
    #         except:
    #             xr_ds1 = xr.merge(output1, combine_attrs="override")
    #     else:
    #         try:
    #             xr_ds1 = xr.combine_by_coords(output1, data_vars='minimal')
    #         except:
    #             xr_ds1 = xr.merge(output1, combine_attrs="override")

    #     ## Output
    #     output3 = process_results_output(xr_ds1, parameter, modified_date, quality_code, output, squeeze_dims)

    #     return output3



######################################
### Testing
