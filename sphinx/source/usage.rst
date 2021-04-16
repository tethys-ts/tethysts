How to use tethysts
=====================

Background
-----------
This section will describe how to use the tethysts package. The functions depend heavily on the xarray package. Nearly all outputs are either as xarray Datasets or dictionaries.

All files/objects in S3 object storage are stored in buckets. You can think of these buckets as root folders housing data files.

The data in those buckets are organised in three layers:
  - Datasets
  - Stations
  - Results

You first need to figure out what datasets exist in each bucket.
Let's say you've got a bucket called fire-emergency-nz and you've been told that the connection_config is https://b2.tethys-ts.xyz:

.. code:: python

    from tethysts import Tethys

    remote = {'connection_config': 'https://b2.tethys-ts.xyz', 'bucket': 'fire-emergency-nz'}


.. ipython:: python
   :suppress:

   from tethysts import Tethys

   remote = {'connection_config': 'https://b2.tethys-ts.xyz', 'bucket': 'fire-emergency-nz'}
   dataset_id = 'dddb02cd5cb7ae191311ab19'
   station_id = 'fedeb59e6c7f47597a7d47c7'


Initialise the class to determine the available datasets:

.. ipython:: python

  t1 = Tethys([remote])
  datasets = t1.datasets
  my_dataset = [d for d in datasets if (d['parameter'] == 'temperature') and (d['aggregation_statistic'] == 'mean')][0]
  print(my_dataset)

In this example we only have one remote we want to check for datasets, but as you can see the initialisation takes a list of remotes (dicts). If you had more remotes, then you would just need to put them all together in a list and pass them to the Tethys class.

The datasets object is a list of dictionaries with a lot of metadata about each dataset. It should tell you practically all you need to know about data contained in the results (e.g. parameter, units, data licence, owner, etc).

Once you've decided which dataset you want (i.e. mean hourly air temperature), write down the dataset_id contained within the associated dictionary and pass it to the next method: get_stations.

.. ipython:: python

  dataset_id = 'dddb02cd5cb7ae191311ab19'

  stations = t1.get_stations(dataset_id)
  my_station = [s for s in stations if (s['ref'] == 'waeranga')][0]
  print(my_station)

Again, the stations object is a list of dictionaries. Most of the data in each dictionary should be self-explanitory. But what you'll need next is to pick a station and write down the station_id just like you did with the dataset_id.

To get the results (the time series data), you'll need a dataset_id and station_id. Internally, the results are broken up by dataset and station.
The get_results method has many input options. Take a look at the reference page for a description of all the options.

.. ipython:: python

  station_id = 'fedeb59e6c7f47597a7d47c7'

  results = t1.get_results(dataset_id, station_id, remove_height=True, output='Dataset')
  print(results)

Unlike the previously returned objects, the results object (in this case) is an xarray Dataset. This xarray Dataset contains both the results (temperature) and all of the previous dataset and station dictionary data. The default output type is an xarray DataArray, which does not contain the station data, but does contain the dataset data. The results are stored/structured according to CF conventions v1.8.
