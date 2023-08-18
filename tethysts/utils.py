"""

"""
import io
import numpy as np
import xarray as xr
import pandas as pd
import orjson
from datetime import datetime
import zstandard as zstd
import copy
import pickle
import botocore
from shapely.geometry import shape, Polygon, Point
from shapely.strtree import STRtree
from typing import Optional, List, Any, Union
from scipy import spatial
import pathlib
from pydantic import HttpUrl
from hdf5tools import H5
import s3tethys

# pd.options.display.max_columns = 10

##############################################
### Reference objects

b2_public_key_pattern = '{base_url}/{bucket}/{obj_key}'
contabo_public_key_pattern = '{base_url}:{bucket}/{obj_key}'
public_remote_key = 'https://b2.tethys-ts.xyz/file/tethysts/tethys/public_remotes_v4.json.zst'

local_results_name = '{ds_id}/{version_date}/{stn_id}/{chunk_id}.{chunk_hash}.results.h5'

s3_url_base = 's3://{bucket}/{key}'

##############################################
### Helper functions


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
    res_index = strtree.nearest(geom_query)

    stn_ids_list = list(stns.keys())
    stn_id = stn_ids_list[res_index]

    return stn_id


def get_intersected_stations(stns, geom_query):
    """

    """
    if isinstance(geom_query, dict):
        geom_query = shape(geom_query)

    stn_ids_list = list(stns.keys())
    geom1 = [shape(s['geometry']) for i, s in stns.items()]
    strtree = STRtree(geom1)
    res_index = strtree.query(geom_query)

    stn_ids = [stn_ids_list[r] for r in res_index]

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


def chunk_filters(results_chunks, stn_ids, time_interval=None, from_date=None, to_date=None, heights=None, bands=None, from_mod_date=None, to_mod_date=None):
    """

    """
    ## Stations filter
    rc2 = copy.deepcopy([rc for rc in results_chunks if rc['station_id'] in stn_ids])
    first_one = rc2[0]

    ## Temporal filters
    if isinstance(from_date, (str, pd.Timestamp, datetime)) and ('chunk_day' in first_one):
        from_date1 = int(pd.Timestamp(from_date).timestamp()/60/60/24)
        rc2 = [rc for rc in rc2 if (rc['chunk_day'] + time_interval) >= from_date1]

    if len(rc2) == 0:
        return rc2

    if isinstance(to_date, (str, pd.Timestamp, datetime)) and ('chunk_day' in first_one):
        to_date1 = int(pd.Timestamp(to_date).timestamp()/60/60/24)
        rc2 = [rc for rc in rc2 if rc['chunk_day'] <= to_date1]

    if len(rc2) == 0:
        return rc2

    if isinstance(from_mod_date, (str, pd.Timestamp, datetime)) and ('modified_date' in first_one):
        from_mod_date1 = pd.Timestamp(from_mod_date)
        rc2 = [rc for rc in rc2 if pd.Timestamp(rc['modified_date']) >= from_mod_date1]

    if len(rc2) == 0:
        return rc2

    if isinstance(to_mod_date, (str, pd.Timestamp, datetime)) and ('modified_date' in first_one):
        to_mod_date1 = pd.Timestamp(to_mod_date)
        rc2 = [rc for rc in rc2 if pd.Timestamp(rc['modified_date']) <= to_mod_date1]

    if len(rc2) == 0:
        return rc2

    ## Heights and bands filter
    if (heights is not None) and ('height' in first_one):
        if isinstance(heights, (int, float)):
            h1 = [int(heights*1000)]
        elif isinstance(heights, list):
            h1 = [int(h*1000) for h in heights]
        else:
            raise TypeError('heights must be an int, float, or list of int/float.')
        rc2 = [rc for rc in rc2 if rc['height'] in h1]

    if len(rc2) == 0:
        return rc2

    if (bands is not None) and ('band' in first_one):
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


def result_filters(h5, from_date=None, to_date=None):
    """

    """
    h5 = h5.sel(exclude_coords=['station_geometry', 'chunk_date'])

    ## Time filters
    if isinstance(from_date, (str, pd.Timestamp, datetime)):
        from_date1 = np.datetime64(from_date)
    else:
        from_date1 = None
    if isinstance(to_date, (str, pd.Timestamp, datetime)):
        to_date1 = np.datetime64(to_date)
    else:
        to_date1 = None

    if (to_date1 is not None) or (from_date1 is not None):
        h5 = h5.sel({'time': slice(from_date1, to_date1)})

    return h5


def download_results(chunk: dict, bucket: str, s3: botocore.client.BaseClient = None, connection_config: dict = None, public_url: HttpUrl = None, cache: Union[pathlib.Path] = None, from_date=None, to_date=None, return_raw=False):
    """

    """
    file_obj = s3tethys.get_object_s3(chunk['key'], bucket, s3, connection_config, public_url)

    if isinstance(cache, pathlib.Path):
        chunk_hash = chunk['chunk_hash']
        version_date = pd.Timestamp(chunk['version_date']).strftime('%Y%m%d%H%M%SZ')
        results_file_name = local_results_name.format(ds_id=chunk['dataset_id'], stn_id=chunk['station_id'], chunk_id=chunk['chunk_id'], version_date=version_date, chunk_hash=chunk_hash)
        chunk_path = cache.joinpath(results_file_name)

        if not chunk_path.exists():
            chunk_path.parent.mkdir(parents=True, exist_ok=True)

            if chunk['key'].endswith('.zst'):
                data = xr.load_dataset(s3tethys.decompress_stream_to_object(io.BytesIO(file_obj.read()), 'zstd'))
                H5(data).sel(exclude_coords=['station_geometry', 'chunk_date']).to_hdf5(chunk_path, compression='zstd')
                data.close()
                del data
            else:
                s3tethys.stream_to_file(file_obj, chunk_path)

        data_obj = chunk_path

    else:
        if return_raw:
            return file_obj

        if chunk['key'].endswith('.zst'):
            file_obj = s3tethys.decompress_stream_to_object(io.BytesIO(file_obj.read()), 'zstd')
            data = xr.load_dataset(file_obj.read(), engine='scipy')
        else:
            data = io.BytesIO(file_obj.read())

        h1 = H5(data)
        data_obj = io.BytesIO()
        h1 = result_filters(h1)
        h1.to_hdf5(data_obj, compression='zstd')

        if isinstance(data, xr.Dataset):
            data.close()
        del data
        del h1

    del file_obj

    return data_obj


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

    # Update the attributes in the coords from the first ds
    for coord in xr3.coords:
        xr3[coord].encoding = datasets[0][coord].encoding
        xr3[coord].attrs = datasets[0][coord].attrs

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


def filter_mod_dates(results, from_mod_date=None, to_mod_date=None):
    """
    Need to do this because xarray "where" is useless...
    """
    if ((from_mod_date is not None) or (to_mod_date is not None)) and ('modified_date' in results):
        mod_dates = results['modified_date'].copy().load()

        if (from_mod_date is not None) and (to_mod_date is not None):
            mod_bool = (mod_dates >= pd.Timestamp(from_mod_date)) & (mod_dates <= pd.Timestamp(to_mod_date))
        elif (from_mod_date is not None):
            mod_bool = (mod_dates >= pd.Timestamp(from_mod_date))
        elif (to_mod_date is not None):
            mod_bool = (mod_dates <= pd.Timestamp(to_mod_date))

        data_vars1 = [var for var in results.data_vars if 'time' in results[var].dims]

        results[data_vars1] = results[data_vars1].where(mod_bool)

        return results.dropna('time', how='all')
    else:
        return results


def results_concat(results_list, output_path=None, from_date=None, to_date=None, from_mod_date=None, to_mod_date=None, compression='lzf'):
    """

    """
    if output_path is None:
        output_path = io.BytesIO()
        compression = 'zstd'

    h1 = H5(results_list)
    h1 = result_filters(h1, from_date, to_date)
    h1.to_hdf5(output_path, compression=compression)

    xr3 = xr.open_dataset(output_path, engine='h5netcdf', cache=False)

    ## Deal with mod dates filters
    if ((from_mod_date is not None) or (to_mod_date is not None)) and ('modified_date' in xr3):
        xr3 = filter_mod_dates(xr3, from_mod_date, to_mod_date)

    return xr3
