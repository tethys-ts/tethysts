How to use tethysts
=====================

Background
-----------
This section will describe how to use the tethysts package. The functions depend heavily on the xarray package. Nearly all outputs are either as xarray Datasets or dictionaries.

The datasets are organised in three layers:
  - Dataset metadata
  - Stations
  - Results

Dataset metadata
----------------
You first need to figure out what datasets exist in each bucket.
Let's say you've got a bucket called ecan-env-monitoring and you've been told that the connection_config is https://b2.tethys-ts.xyz:

.. code:: python

    from tethysts import Tethys

    remote = {'bucket': 'ecan-env-monitoring', 'connection_config': 'https://b2.tethys-ts.xyz'}


.. ipython:: python
   :suppress:

   from tethysts import Tethys
   from pprint import pprint as print

   remote = {'bucket': 'ecan-env-monitoring', 'connection_config': 'https://b2.tethys-ts.xyz'}
   dataset_id = 'b5d84aa773de2a747079c127'
   station_id = '4db28a9db0cb036507490887'


Initialise the class to get the metadata about the available datasets:

.. ipython:: python

  ts = Tethys([remote])
  datasets = ts.datasets
  my_dataset = [d for d in datasets if (d['parameter'] == 'precipitation') and
                                       (d['product_code'] == 'raw_data') and
                                       (d['frequency_interval'] == '24H') and
                                       (d['owner'] == 'Environment Canterbury')][0]
  my_dataset

In this example we only have one remote we want to check for datasets, but as you can see the initialisation takes a list of remotes (dicts). If you had more remotes, then you would just need to put them all together in a list and pass them to the Tethys class.

Alternatively, from version 0.2.7 if you do not have specific remotes to pass to the Tethys object, then you can simply initialise Tethys without anything (None) and Tethys will pull down all public remotes and parse the associated dataset metadata:

.. ipython:: python

  ts = Tethys()
  datasets = ts.datasets
  my_dataset = [d for d in datasets if (d['parameter'] == 'precipitation') and
                                       (d['product_code'] == 'raw_data') and
                                       (d['frequency_interval'] == '24H') and
                                       (d['owner'] == 'Environment Canterbury')][0]
  my_dataset

The datasets object is a list of dictionaries with a lot of metadata about each dataset. It should tell you practically all you need to know about data contained in the results (e.g. parameter, units, data licence, owner, etc).

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

  dataset_id = 'b5d84aa773de2a747079c127'
  geometry = {'type': 'Point', 'coordinates': [172.0, -42.8]}

  my_station = ts.get_stations(dataset_id, geometry=geometry)
  my_station[0]

To get a bunch of stations within a specified area, you can pass a polygon GeoJSON geometry or a combination of latitude, longitude, and distance (radius in decimal degrees).

.. ipython:: python

  dataset_id = 'b5d84aa773de2a747079c127'
  lon = 172.0
  lat = -42.8
  distance = 0.2

  my_stations = ts.get_stations(dataset_id, lat=lat, lon=lon, distance=distance)
  my_stations

Results
-------
But what you'll need next is to pick a station and write down the station_id just like you did with the dataset_id.

To get the results (the time series data), you'll need a dataset_id and station_id. Internally, the results are broken up by dataset and station.
The get_results method has many input options. Take a look at the reference page for a description of all the options.

.. ipython:: python

  station_id = '4db28a9db0cb036507490887'

  results = ts.get_results(dataset_id, station_id, output='Dataset')
  results

Unlike the previously returned objects, the results object (in this case) is an xarray Dataset. This xarray Dataset contains both the results (temperature) and all of the previous dataset and station data. Other options include an xarray DataArray, dictionary, and JSON. The results are indexed by geometry, height, and time. The geometry dimension is a hexadecimal encoded Well-Known Binary (WKB) representation of the geometry. This was used to be flexible on the geometry type (i.e. points, lines, or polygons) and the WKB ensures that the geometry is stored accurately. This is a standard format by the Open Geospatial Consortium (OGC) and can be parsed by many programs including shapely, PostGIS, etc. Using WKB in a geometry dimension does not follow CF conventions. This was a trade off between flexibility, simplicity, and following standards. I picked flexibility and simplicity.

Similar to the get_stations spatial query, the get_results method has a built-in nearest neighbour query if you omit the station_id and pass either geometry dict or a combination of latitude and longitude.

.. ipython:: python

  station_id = '4db28a9db0cb036507490887'
  geometry = {'type': 'Point', 'coordinates': [172.0, -42.8]}

  results = ts.get_results(dataset_id, geometry=geometry, squeeze_dims=True, output='Dataset')
  results

If you want to get more than one station per dataset, then you can use the get_bulk_results. This simply runs concurrent thread requests for multiple stations results. The output will concatenate on the geometry dimension.

.. ipython:: python

  station_ids = [station_id, '474f75b4de127caca088620a']

  results = ts.get_bulk_results(dataset_id, station_ids, squeeze_dims=True, output='Dataset')
  results

If a run_date is not passed to the get_results method, then the latest run date will be returned. If you'd like to list all the run dates and to choose which run date you'd like to pass to the get_results or get_bulk_results methods, then you can use the get_run_dates method.

.. ipython:: python

  run_dates = ts.get_run_dates(dataset_id, station_id)
  run_dates


Tethys web API
--------------
The `Tethys web API <https://api.tethys-ts.xyz/docs>`_ uses all of the same function names and associated input parameters as the Python package. But in most cases, users should use the Python package instead of the web API as it will be faster, more flexible, and won't put load on the VM running the web API.
