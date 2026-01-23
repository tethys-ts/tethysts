Installation
============
Install via pip::

  pip install tethysts==4.5.16 hdf5tools==0.2.4

Or conda::

  conda install -c mullenkamp tethysts=4.5.16 hdf5tools=0.2.4

Requirements
------------
The package dependencies are: tethys-data-models, zstandard, pandas, xarray, scipy, boto3, orjson, requests, shapely, s3tethys, and hdf5tools.

Important note
---------------
This package as well as other dependencies are continuously being updated and improved. I am also updating and maintaining this package in my spare time. This does mean that I may accidentally cause some breaking changes to old versions across dependencies. If you find yourself with some weird error, try to update the following packages to the most recent version:

  - tethysts
  - hdf5tools
  - shapely
