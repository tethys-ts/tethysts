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
# da2.encoding = {'dtype': 'int16', '_FillValue': -9999, 'scale_factor': 1}
# da2.encoding = {'dtype': 'int16', '_FillValue': -9999, 'scale_factor': 0.1}
da2.encoding = {}

ds1 = da2.to_dataset()

ds1['time'] = pd.to_datetime(['2000-01-01'])
ds1 = ds1.squeeze('time')
ds1 = ds1.expand_dims('time')

da3 = ds1.altitude
arr = da3

# nc1 = tu.write_pkl_zstd(ds1.to_netcdf())
# len(nc1)

# nc2 = xr.load_dataset(tu.read_pkl_zstd(nc1))

# ar1 = np.array_split(da2, [100, 100])

# ar1 = da2.values

# ar2 = np.array_split(ar1, [100, 100], axis=1)


def split_grid(arr, x_size, y_size, x_name='x', y_name='y'):
    """
    Function to split an n-dimensional dataset along the x and y dimensions.

    Parameters
    ----------
    arr : DataArray
        An xarray DataArray with at least x and y dimensions. It can have any number of dimensions, though it probably does not make much sense to have greater than 4 dimensions.
    x_size : int
        The size or length of the smaller grids in the x dimension.
    y_size : int
        The size or length of the smaller grids in the y dimension.
    x_name : str
        The x dimension name.
    y_name : str
        The y dimension name.

    Returns
    -------
    List of DataArrays
        The result contains none of the original attributes.
    """
    ## Get the dimension data
    dims = arr.dims
    x_index = dims.index(x_name)
    y_index = dims.index(y_name)
    data_name = arr.name

    arr_shape = arr.shape

    m = arr_shape[x_index]
    n = arr_shape[y_index]
    dtype = arr.dtype

    ## Build the new regular array to be queried
    y_diff = arr[y_name].diff(y_name, 1).median().values
    x_diff = arr[x_name].diff(x_name, 1).median().values

    bpx = ((m-1)//x_size + 1) # blocks per x
    bpy = ((n-1)//y_size + 1) # blocks per y
    M = x_size * bpx
    N = y_size * bpy

    x_y = list(arr_shape)
    x_y[x_index] = M
    x_y[y_index] = N

    sel1 = tuple(slice(0, s) for s in arr_shape)

    A = np.nan * np.ones(x_y)
    A[sel1] = arr

    # x array
    x_start = arr[x_name][0].values
    x_int = M * x_diff
    x_end = x_start + x_int
    xs = np.arange(x_start, x_end, x_diff)

    # y array
    y_start = arr[y_name][0].values
    y_int = M * y_diff
    y_end = y_start + y_int
    ys = np.arange(y_start, y_end, y_diff)

    # Coords
    coords = []
    new_dims = []
    for d in dims:
        name = d
        if d == x_name:
            c = xs
        elif d == y_name:
            c = ys
        else:
            c = arr[d]
        coords.extend([c])
        new_dims.extend([name])

    # New DataArray
    A1 = xr.DataArray(A, coords=coords, dims=new_dims, name=data_name)

    block_list = []
    previous_x = 0
    for x_block in range(bpy):
        previous_x = x_block * x_size
        previous_y = 0
        for y_block in range(bpx):
            previous_y = y_block * y_size
            x_slice = slice(previous_x, previous_x+x_size)
            y_slice = slice(previous_y, previous_y+y_size)

            sel2 = list(sel1)
            sel2[x_index] = x_slice
            sel2[y_index] = y_slice

            block = A1[tuple(sel2)]

            # remove nan ys and nan xs
            # nan_y = np.all(np.isnan(block), axis=y_index)
            block = block.dropna(y_name, 'all')
            # nan_x = np.all(np.isnan(block), axis=x_index)
            block = block.dropna(x_name, 'all')

            ## append
            if block.size:
                block_list.append(block.astype(dtype))

    return block_list


def determine_array_size(arr, starting_x_size=100, starting_y_size=100, increment=100, min_size=800, max_size=1100, x_name='x', y_name='y'):
    """
    Function to determine the appropriate grid size for splitting.

    Parameters
    ----------
    arr : DataArray
        An xarray DataArray with at least x and y dimensions. It can have any number of dimensions, though it probably does not make much sense to have greater than 4 dimensions.
    starting_x_size : int
        The initial size or length of the smaller grids in the x dimension.
    starting_y_size : int
        The initial size or length of the smaller grids in the y dimension.
    increment : int
        The incremental grid size to be added iteratively to the starting sizes.
    min_size : int
        The minimum acceptable object size in KB.
    max_size : int
        The maximum acceptable object size in KB.
    x_name : str
        The x dimension name.
    y_name : str
        The y dimension name.

    Returns
    -------
    dict
        Of the optimised grid size results.
    """
    max_obj_size = 0
    x_size = starting_x_size
    y_size = starting_y_size

    while True:
        block_list = split_grid(arr, x_size=x_size, y_size=y_size, x_name=x_name, y_name=y_name)
        obj_sizes = [len(tu.write_pkl_zstd(nc.to_netcdf())) for nc in block_list]
        max_obj_size = max(obj_sizes)

        if max_obj_size < min_size*1000:
            x_size = x_size + increment
            y_size = y_size + increment
        else:
            break

    if max_obj_size > max_size*1000:
        print('max_object_size:', str(max_obj_size))
        raise ValueError('max object size is greater than the allotted size. Reduce the increment value and start again.')

    obj_dict = {'x_size': x_size, 'y_size': y_size, 'max_obj_size': max_obj_size, 'min_obj_size': min(obj_sizes)}

    return obj_dict


obj_dict = determine_array_size(da3, 600, 600, max_size=1100000)

block_list = split_grid(da3, obj_dict['x_size'], obj_dict['y_size'])

obj_sizes = [len(tu.write_pkl_zstd(nc.to_netcdf())) for nc in block_list]

max(obj_sizes)
min(obj_sizes)



shape1 = [nc.shape for nc in block_list]

shape2 = [nc.shape[2] * nc.shape[1] for nc in block_list]

ds1 = [da.to_dataset() for da in block_list]


# xr.combine_by_coords(ar4)
ds2 = xr.combine_by_coords(ds1)


b1 = 'l49f0sje'.encode('utf-8')

h1 = b1.hex()

p1 = Point(4, 89)

b2 = p1.wkb_hex





