# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:25:41 2019

@author: michaelek
"""
import os
import pandas as pd
import xarray as xr
import tethys_utils as tu
import numpy as np

pd.options.display.max_columns = 10

#################################################
### Parameters

tif1 = '/media/sdb1/Data/nasa/aster/dem/ASTGTMV003_S45E168_dem.tif'








################################################
### Splitting gridded data

da1 = xr.open_rasterio(tif1)
da2 = da1.squeeze('band').drop('band')
da2.attrs = {}
da2.name = 'altitude'

# da2.encoding = {'dtype': 'int32', '_FillValue': -9999, 'scale_factor': 0.1}
da2.encoding = {'dtype': 'int16', '_FillValue': -9999, 'scale_factor': 1}
# da2.encoding = {'dtype': 'int16', '_FillValue': -9999, 'scale_factor': 0.1}
da2.encoding = {}

ds1 = da2.to_dataset()

ds1['time'] = pd.to_datetime(['2000-01-01'])
ds1 = ds1.squeeze('time')
ds1 = ds1.expand_dims('time')

da3 = ds1.altitude
arr = da3

nc1 = tu.write_pkl_zstd(ds1.to_netcdf())
len(nc1)

nc2 = xr.load_dataset(tu.read_pkl_zstd(nc1))

ar1 = np.array_split(da2, [100, 100])

ar1 = da2.values

ar2 = np.array_split(ar1, [100, 100], axis=1)


def split_grid(arr, lon_size, lat_size, lon_name='x', lat_name='y', data_name='altitude'):
    """

    """
    dims = arr.dims
    lon_index = dims.index(lon_name)
    lat_index = dims.index(lat_name)

    arr_shape = arr.shape

    m = arr_shape[lon_index]
    n = arr_shape[lat_index]
    dtype = arr.dtype

    ## Build the new even array to be queried
    lat_diff = arr[lat_name].diff(lat_name, 1).median().values
    lon_diff = arr[lon_name].diff(lon_name, 1).median().values

    bpr = ((m-1)//lon_size + 1) # blocks per lon
    bpc = ((n-1)//lat_size + 1) # blocks per lat
    M = lon_size * bpr
    N = lat_size * bpc

    lon_lat = list(arr_shape)
    lon_lat[lon_index] = M
    lon_lat[lat_index] = N

    sel1 = tuple(slice(0, s) for s in arr_shape)

    A = np.nan * np.ones(lon_lat)
    A[sel1] = arr

    # Lon array
    lon_start = arr[lon_name][0].values
    lon_int = M * lon_diff
    lon_end = lon_start + lon_int
    lons = np.arange(lon_start, lon_end, lon_diff)

    # Lat array
    lat_start = arr[lat_name][0].values
    lat_int = M * lat_diff
    lat_end = lat_start + lat_int
    lats = np.arange(lat_start, lat_end, lat_diff)

    # Coords
    coords = []
    new_dims = []
    for d in dims:
        if d == lon_name:
            c = lons
            name = 'lon'
        elif d == lat_name:
            c = lats
            name = 'lat'
        else:
            c = arr[d]
            name = d
        coords.extend([c])
        new_dims.extend([name])

    # New DataArray
    A1 = xr.DataArray(A, coords=coords, dims=new_dims, name=data_name)

    block_list = []
    previous_lon = 0
    for lon_block in range(bpc):
        previous_lon = lon_block * lon_size
        previous_lat = 0
        for lat_block in range(bpr):
            previous_lat = lat_block * lat_size
            lon_slice = slice(previous_lon, previous_lon+lon_size)
            lat_slice = slice(previous_lat, previous_lat+lat_size)

            sel2 = list(sel1)
            sel2[lon_index] = lon_slice
            sel2[lat_index] = lat_slice

            block = A1[tuple(sel2)]

            # remove nan lats and nan lons
            # nan_lat = np.all(np.isnan(block), axis=lat_index)
            block = block.dropna('lat', 'all')
            # nan_lon = np.all(np.isnan(block), axis=lon_index)
            block = block.dropna('lon', 'all')

            ## append
            if block.size:
                block_list.append(block.astype(dtype))

    return block_list


def determine_array_size(arr, starting_lon_size=100, starting_lat_size=100, increment=100, min_size=800, max_size=1100):
    """
    Function to split a gridded DataArray into many smaller gridded DataArrays.

    """
    max_obj_size = 0

    lon_size = starting_lon_size
    lat_size = starting_lat_size

    while True:
        block_list = split_grid(arr, lon_size=lon_size, lat_size=lat_size)
        obj_sizes = [len(tu.write_pkl_zstd(nc.to_netcdf())) for nc in block_list]
        max_obj_size = max(obj_sizes)

        if max_obj_size < min_size*1000:
            lon_size = lon_size + increment
            lat_size = lat_size + increment
        else:
            break

    if max_obj_size > max_size*1000:
        print(str(max_obj_size))
        raise ValueError('max object size is greater than the allotted size. Reduce the increment value and start again.')

    obj_dict = {'lon_size': lon_size, 'lat_size': lat_size, 'max_obj_size': max_obj_size, 'min_obj_size': min(obj_sizes)}

    return obj_dict


obj_dict = determine_array_size(da2, 600, 600, max_size=1100000)

block_list = split_grid(da3, 800, 800)

obj_sizes = [len(tu.write_pkl_zstd(nc.to_netcdf())) for nc in block_list]

max(obj_sizes)
min(obj_sizes)



shape1 = [nc.shape for nc in block_list]

shape2 = [nc.shape[2] * nc.shape[1] for nc in block_list]

ds1 = [da.to_dataset() for da in block_list]


# xr.combine_by_coords(ar4)
ds2 = xr.combine_by_coords(ds1)













