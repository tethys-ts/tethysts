Basic usage
=====================

Background
-----------
This section describes how to use the tethysts package. The functions depend heavily on the `xarray package <http://xarray.pydata.org/>`_. Nearly all outputs are either as xarray Datasets or python lists of dictionaries.

The datasets are organised in three main layers:
  - Dataset metadata
  - Stations
  - Results

There is also versioning of the Stations and Results. Dataset metadata is not currently versioned.

Dataset metadata
----------------
The first step is to figure out what datasets are available.
Import the Tethys class:


.. ipython:: python
   :suppress:
   :okwarning:

   from tethysts import Tethys
   from pprint import pprint as print
   import warnings
   warnings.filterwarnings("ignore")

   remotes = [{'bucket': 'ecan-env-monitoring', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 4}]
   dataset_id = 'b5d84aa773de2a747079c127'
   station_id = 'f9c61373e7ca386c1fab06db'

.. code:: python

   from tethysts import Tethys


Public datasets
~~~~~~~~~~~~~~~
Initialising the Tethys class without any parameters will pull down all public remotes and parse the associated dataset metadata. The datasets object is a list of dictionaries with a lot of metadata about each dataset. It should tell you practically all you need to know about the data contained in the results (e.g. parameter, units, data licence, owner, etc). Use normal python list comprehension to select the dataset(s) of interest:


.. ipython:: python

  ts = Tethys()
  datasets = ts.datasets
  my_dataset = [d for d in datasets if (d['parameter'] == 'precipitation') and
                                       (d['product_code'] == 'raw_data') and
                                       (d['frequency_interval'] == '24H') and
                                       (d['owner'] == 'Environment Canterbury')][0]
  my_dataset


Private datasets
~~~~~~~~~~~~~~~~
Some datasets are not available through the public repository. Accessing private datasets stored in S3 buckets requires remote connection configuration data. A remote configuration requires a list of dictionaries of bucket name, connection_config/public_url, and system version as shown in the following example:


.. code:: python

    from tethysts import Tethys

    remotes = [{'bucket': 'ecan-env-monitoring', 'public_url': 'https://b2.tethys-ts.xyz/file/', 'version': 4}]


Initialise the class with the remotes to get the metadata about the available datasets:

.. ipython:: python

  ts = Tethys(remotes)
  datasets = ts.datasets
  my_dataset = [d for d in datasets if (d['parameter'] == 'precipitation') and
                                       (d['product_code'] == 'raw_data') and
                                       (d['frequency_interval'] == '24H') and
                                       (d['owner'] == 'Environment Canterbury')][0]
  my_dataset

In this example there is one remote we want to check for datasets, but more dictionaries can be added to the remotes list to parse more datasets.

Caching
~~~~~~~~~~~~~~~~
New in version 4, the Tethys class can now be initialized with a local cache path. Tethys can now download the results chunks locally to be used again in future get_results calls. This is generally recommended.

Just pass a cache path when Tethys is initialized:

.. code:: python

    from tethysts import Tethys

    ts = Tethys(remotes, cache='/my/local/cache/path')


Stations
--------
Once you've decided which dataset you want (i.e. cumulative 24 hour precipitation), write down the dataset_id contained within the associated dictionary and pass it to the next method: get_stations.

.. ipython:: python

  dataset_id = 'b5d84aa773de2a747079c127'

  stations = ts.get_stations(dataset_id)
  my_station = [s for s in stations if (s['name'] == "Waimakariri at Arthur's Pass")][0]
  my_station


Again, the stations object is a list of dictionaries. Most of the data in each dictionary should be self-explanatory.

If you've got geographic coordinates as a GeoJSON point or a combination of a latitude and longitude, then this can be passed to the get_stations method to get the nearest single station.

.. ipython:: python
  :okwarning:

  dataset_id = 'b5d84aa773de2a747079c127'
  geometry = {'type': 'Point', 'coordinates': [172.0, -42.8]}

  my_station = ts.get_stations(dataset_id, geometry=geometry)
  my_station[0]


To get a bunch of stations within a specified area, you can pass a polygon GeoJSON geometry or a combination of latitude, longitude, and distance (radius in decimal degrees).

.. ipython:: python
  :okwarning:

  dataset_id = 'b5d84aa773de2a747079c127'
  lon = 172.0
  lat = -42.8
  distance = 0.2

  my_stations = ts.get_stations(dataset_id, lat=lat, lon=lon, distance=distance)
  my_stations


Results
-------
What you'll need next is to pick a station and write down the station_id just like you did with the dataset_id.

To get the results (the 4D data), you'll need a dataset_id and station_id. Internally, the results are broken up by dataset and station.
The get_results method has many input options. Take a look at the reference page for a description of all the options.

.. ipython:: python

  station_id = 'f9c61373e7ca386c1fab06db'

  results = ts.get_results(dataset_id, station_id)
  results

Unlike the previously returned objects, the results object (in this case) is an xarray Dataset. This xarray Dataset contains both the results (temperature) and all of the dataset metadata. If the results represent geospatially sparse data, then the results are indexed by geometry, height, and time. If the results represent gridded data, then the results are indexed by lat, lon, height, and time. The geometry dimension is a hexadecimal encoded Well-Known Binary (WKB) representation of the geometry. This was used to be flexible on the geometry type (i.e. points, lines, or polygons) and the WKB ensures that the geometry is stored accurately. This is a standard format by the Open Geospatial Consortium (OGC) and can be parsed by many programs including shapely, PostGIS, etc. Using WKB in a geometry dimension does not follow CF conventions, however. This was a trade off between flexibility, simplicity, and following standards. I leaned towards flexibility and simplicity on this one.

In addition to the get_stations spatial queries, the get_results method has a built-in nearest neighbour query if you omit the station_id and pass either geometry dict or a combination of latitude and longitude. This is especially useful for gridded results when each station represents a large area rather than a single point.

.. ipython:: python
  :okwarning:

  geometry = {'type': 'Point', 'coordinates': [172.0, -42.8]}

  results = ts.get_results(dataset_id, geometry=geometry, squeeze_dims=True)
  results

If you want to get more than one station per dataset, then you can still use the get_results. The output will concatenate the xarray Datasets together and return a single xarray Dataset. Since the get_results method is multithreaded when downloading results, passing multiple station ids to it will be much faster than using a "for loop" over each station id.

.. ipython:: python

  station_ids = [station_id, '96e9ff9437fc738b24d10b42']

  results = ts.get_results(dataset_id, station_ids)
  results


Saving to hdf5 files
~~~~~~~~~~~~~~~~~~~~
Starting in version 4.5, Tethys can now save results directly to hdf5 files that can be opened by xarray. You must specify an output_path and optionally a compression for the hdf5 file (lzf is the default compression). There's no concern for excessive data volume in this process. You can download results from one station or all stations in a dataset to a single file without much trouble. It's recommended to save the file with the .h5 extension rather than the .nc extension to make it clear that it's a normal hdf5 file rather than a fully netcdf4-compliant file. Future versions might be formatted to be fully netcdf4-compliant...if I can figure out all of the nuances...any help is appreciated! *Update using hdf5tools>=0.1.12*...I've managed to make the hdf5 file compatible with the python netcdf4 package. This means that files created by the tethysts package should be compatible with any python packages that read netcdf4 data (which of course includes xarray).

.. code:: python

    results = ts.get_results(dataset_id, station_ids, output_path='/my/local/path/results.h5', compression='lzf')


And if you'd like to reopen the hdf5 file with xarray later, then you can use the xr.open_dataset function as normal (even with advanced compression...somehow...).

.. code:: python

    results = xr.open_dataset('/my/local/path/results.h5')


Selective filters
~~~~~~~~~~~~~~~~~
In Tethys version 4, the results have been saved into multiple chunks. These chunks contain specific time periods, heights, and stations. It is best to provide from_date, to_date, and heights filters to the get_results method so that less data needs to be downloaded and concatenated. If you don't, you might end up downloading a lot of data, using a lot of RAM, and consuming a lot of processing time unnecessarily.

Dataset versions
----------------
If a version_date is not passed to the get_results or get_stations method, then the latest dataset version will be returned. If you'd like to list all the dataset versions and to choose which version you'd like to pass to the get_results or get_stations method, then you can use the get_versions method.

.. ipython:: python

  versions = ts.get_versions(dataset_id)
  versions

Handling geometries
---------------------
Depending data request, Tethys will either return geometries as GeoJSON or Well-Known Binary (WKB) hexadecimal geometries. If you're not familiar with how to handle these formats, I recommend using `Shapely <https://shapely.readthedocs.io/en/stable/manual.html>`_ to convert into and out of geometry formats and to provide a range of geospatial processing tools. Shapely is used as the geoprocessing tool behind `geopandas <https://geopandas.org/en/stable/>`_.

For example if you've made a get_stations request and returned GeoJSON geometries, then you could convert them to shapely objects and put them into a dictionary with station_ids as keys:

.. ipython:: python

  from shapely.geometry import shape

  dataset_id = 'b5d84aa773de2a747079c127'

  stations = ts.get_stations(dataset_id)
  stns_geo = {s['station_id']: shape(s['geometry']) for s in stations}
  stns_geo['f9c61373e7ca386c1fab06db']

Or you could convert the WKB hex of results into a list of shapely objects:


.. ipython:: python

  from shapely import wkb

  station_ids = [station_id, '96e9ff9437fc738b24d10b42']

  results = ts.get_results(dataset_id, station_ids)
  geo_list = [wkb.loads(g, hex=True) for g in results.geometry.values]
  geo_list


.. Tethys web API
.. --------------
.. The `Tethys web API <https://api.tethys-ts.xyz/docs>`_ uses all of the same function names and associated input parameters as the Python package. But in most cases, users should use the Python package instead of the web API as it will be faster, more flexible, and won't put load on the VM running the web API.
