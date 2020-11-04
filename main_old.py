# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:56:21 2020

@author: MichaelEK
"""
import sys
import os
import io
import numpy as np
# base_dir = os.path.realpath(os.path.dirname(__file__))
# sys.path.append(base_dir)
import xarray as xr
from pymongo import MongoClient, InsertOne, DeleteOne, ReplaceOne, UpdateOne
from pymongo.errors import BulkWriteError
import pandas as pd
import yaml
from time import sleep
import copy
import boto3
import botocore
from multiprocessing.pool import ThreadPool
from bson.objectid import ObjectId
from tethys_utils import read_pkl_zstd, list_parse_s3, get_last_date, ts_key_pattern, s3_connection
from utils import get_remote_datasets, get_dataset_params
import schedule


#########################################
### Get todays date-time

pd.options.display.max_columns = 10
#run_time_start = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S')
#print(run_time_start)

########################################
### Read in parameters
print('---Read in parameters')

base_dir = os.path.realpath(os.path.dirname(__file__))

# try:
#     param = os.environ.copy()
#     database = param['DATABASE']
#     rw_username = param['RW_USERNAME']
#     rw_password = param['RW_PASSWORD']
# except:
#     with open(os.path.join(base_dir, 'parameters.yml')) as param:
#         param = yaml.safe_load(param)
#
#     database = param['DATABASE']
#     rw_username = param['RW_USERNAME']
#     rw_password = param['RW_PASSWORD']

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)

# base_dir = os.path.realpath(os.path.dirname(__file__))
#
# with open(os.path.join(base_dir, 'datasets.yml')) as param:
#    param = yaml.safe_load(param)

# parser = argparse.ArgumentParser()
# parser.add_argument('yaml_path')
# args = parser.parse_args()
#
# with open(args.yaml_path) as param:
#     param = yaml.safe_load(param)

# datasets_detailed = copy.deepcopy(param['datasets'])
# datasets = copy.deepcopy(datasets_detailed)
#
# field_removal = ['properties', 'units', 'license', 'result_type', 'cf_standard_name', 'precision']
# for ds in datasets:
#     [ds.pop(f) for f in field_removal]


## Process remote datasets

requested_datasets, requested_sites, scheduling, from_mod_date, extra_remotes = get_dataset_params(base_dir)

if isinstance(extra_remotes, list):
    param['remotes'].extend(extra_remotes)

datasets_detailed = get_remote_datasets(param)
datasets = copy.deepcopy(datasets_detailed)
[d.pop('properties') for d in datasets]

db_dict = param['global']['db']

## For testing
# db_dict.update({'HOST': '127.0.0.1'})
# db_dict.update({'HOST': 'tethys-ts.xyz'})


########################################
### Functions


def get_collection_schema_fields(db, collection):
    """

    """
    out1 = db.command({'listCollections': 1, 'filter': {'name': collection}})
    fields = list(out1['cursor']['firstBatch'][0]['options']['validator']['$jsonSchema']['properties'].keys())

    return fields


def get_remote_dataset_properties(dataset):
    """

    """
    ds_index = datasets.index(dataset)
    if ds_index == -1:
        print(str(dataset) + ' is not available in remotes. Please check the dataset codes!!!')
        return None

    t_dataset = copy.deepcopy(datasets_detailed[ds_index])

    prop = t_dataset['properties']

    return prop


def get_ts_key_pattern(dataset):
    """

    """
    prop = get_remote_dataset_properties(dataset)
    ts_key_pattern1 = ts_key_pattern[prop['time_series_type']]

    return ts_key_pattern1


def get_remote_keys(dataset, start_after='', from_key_date=None):
    """

    """
    prop = get_remote_dataset_properties(dataset)

    s3_config = prop['connection_config'].copy()

    s3 = s3_connection(s3_config)

    ts_key_pattern1 = get_ts_key_pattern(dataset)

    skp1 = ts_key_pattern1.split('{date}')[0].format(**dataset)
    df1 = list_parse_s3(s3, prop['bucket'], skp1, start_after=start_after)

    if isinstance(from_key_date, (str, pd.Timestamp)):
        from_key_date1 = pd.Timestamp(from_key_date)
        if from_key_date1.tzname() is None:
            raise ValueError('from_key_date must contain time zone info in some way')
        if from_key_date1.tzname() != 'UTC':
            from_key_date1 = from_key_date1.tz_convert('UTC')
        df1 = df1[df1.KeyDate > from_key_date1].copy()

    if not df1.empty:
        df1 = df1[df1.Key.str.endswith('.nc') | df1.Key.str.endswith('.zst')].copy()

    return df1


def get_db_dataset_id(dataset):
    """

    """
    client = MongoClient(db_dict['HOST'], password=db_dict['RW_PASSWORD'], username=db_dict['RW_USERNAME'], authSource=db_dict['DATABASE'])
    db = client[db_dict['DATABASE']]
    ds_coll = db['dataset']
    try:
        ds1 = ds_coll.find(dataset).limit(1)[0]
    except:
        ds1 = {}

    return ds1


def get_site_ids(sites):
    """

    """
    client = MongoClient(db_dict['HOST'], password=db_dict['RW_PASSWORD'], username=db_dict['RW_USERNAME'], authSource=db_dict['DATABASE'])
    db = client[db_dict['DATABASE']]
    site_coll = db['sampling_site']

    if isinstance(sites, str):
        sites = [sites]

    sites1 = list(site_coll.find({'ref': {'$in': sites}}, {'ref': 1}))

    return sites1


def get_site_dataset_source(dataset, sites=None):
    """

    """
    base_columns = ['Key', 'LastModified', 'ETag', 'Size', 'KeyDate']

    client = MongoClient(db_dict['HOST'], password=db_dict['RW_PASSWORD'], username=db_dict['RW_USERNAME'], authSource=db_dict['DATABASE'])
    db = client[db_dict['DATABASE']]
    ds_coll = db['dataset']
    ds1 = ds_coll.find_one(dataset)
    if ds1 is None:
        print('No dataset found in database; it will be created')
        dataset2 = copy.deepcopy(dataset)
        insert1 = ds_coll.insert_one(dataset2)
        ds_id = dataset2.pop('_id')

        db_keys = pd.DataFrame(columns=base_columns)
    else:
        ds_id = ds1['_id']
        site_ds_coll = db['site_dataset']
        try:
            source_last = list(site_ds_coll.find({'dataset_id': ds_id}, {'source': 1}).sort([('source.LastModified', -1)]).limit(1))[0]['source']
            db_keys = pd.DataFrame([source_last])
        except:
            db_keys = pd.DataFrame(columns=base_columns)

    return db_keys, ds_id


def get_modified_date(date_str):
    """

    """
    date1 = pd.Timestamp(date_str.split(':')[0])

    return date1


def download_objects(grp, s3, bucket, ds_dict):
    """

    """
    print(grp.Key)

    # Download the ts data
    counter = 5
    while counter > 0:
        try:
            obj1 = io.BytesIO()
            s3.download_fileobj(Bucket=bucket, Fileobj=obj1, Key=grp.Key)
            break
        except:
            print('Download failed, will try again in 10 seconds')
            sleep(10)
            counter = counter - 1

    obj1.seek(0)
    if grp.Key.endswith('.zst'):
        obj2 = read_pkl_zstd(obj1.read(), False)
        ds4 = xr.open_dataset(obj2)
    elif grp.Key.endswith('.nc'):
        obj1_read = obj1.read()
        try:
            ds4 = xr.open_dataset(obj1_read)
        except:
            obj2 = read_pkl_zstd(obj1_read, False)
            ds4 = xr.open_dataset(obj2)
    else:
        raise ValueError('Object Key has an unknown extension')

    ds_dict.update({grp.Key: ds4})


def process_site_data(ds4, db, dim_name, results_cols):
    """

    """
    sites_col = ['station_id', 'lat', 'lon']

    extra_sites_cols = list(ds4.variables)
    for c in list(ds4.variables):
        if dim_name == c:
            extra_sites_cols.remove(c)
        elif c in results_cols:
            extra_sites_cols.remove(c)
        elif c in sites_col:
            extra_sites_cols.remove(c)
        elif c == 'time':
            extra_sites_cols.remove(c)

    # Build site document
    site_id = str(ds4['station_id'].values)
    site_coll = db['sampling_site']
    site_geo = {'type': 'Point', 'coordinates': [float(ds4['lon'].values), float(ds4['lat'].values)]}
    site_dict = {'ref': site_id, 'geometry': site_geo, 'virtual_site': bool(ds4['station_id'].attrs['virtual_station'])}
    if 'station_name' in extra_sites_cols:
        site_dict.update({'name': str(ds4['station_name'].values)})
        extra_sites_cols.remove('station_name')
    if extra_sites_cols:
        site_dict.update({'properties': {}})
        for e in extra_sites_cols:
            d1 = ds4[e]
            site_dict['properties'].update({e: {'value': d1.values.tolist(), 'attributes': d1.attrs}})

    # Check that the site has been added and add/update if necessary
    # site_filter = {'ref': site_id, 'geometry': site_geo}
    site_filter = {'geometry': site_geo}
    sites1 = site_coll.find_one(site_filter, {'modified_date': 0})

    if sites1 is None:
        site_dict.update({'modified_date': get_modified_date(ds4.attrs['history'])})
        s1 = site_coll.insert_one(site_dict)
        db_site_id = s1.inserted_id
    else:
        db_site_id = sites1.pop('_id')
        if sites1 != site_dict:
            site_dict.update({'modified_date': get_modified_date(ds4.attrs['history'])})
            site_coll.update_one(site_filter, {'$set': site_dict})

    print('Processing ' + str(db_site_id) + ' with a ref: ' + site_id)

    return db_site_id


def process_dataset_data(db, ds_id, ds4, param_name, ds_fields):
    """

    """
    param_attrs = {k: v for k, v in copy.deepcopy(ds4[param_name].attrs).items() if k in ds_fields}

    ds_coll = db['dataset']
    ds_orig = ds_coll.find_one({'_id': ds_id}, {'_id': 0})

    if ds_orig != param_attrs:
        ds_coll.update_one({'_id': ds_id}, {'$set': param_attrs})


def process_data_H23(ds4, dataset, ds_id, db, source_dict, ds_fields):
    """

    """
    # Get base parameters
    dim_name = list(ds4.dims)[0]
    param_name = dataset['parameter']
    result_type = ds4[param_name].attrs['result_type']

    # Determine the site parameters from the result parameters
    results_cols = [param_name]

    # Check for extra parameters
    param_attrs = ds4[param_name].attrs.copy()
    if 'ancillary_variables' in param_attrs:
        av = param_attrs['ancillary_variables'].split(' ')
        results_cols.extend(av)
    else:
        av = []

    # Process the dataset fields
    process_dataset_data(db, ds_id, ds4, param_name, ds_fields)

    # Process site data
    db_site_id = process_site_data(ds4, db, dim_name, results_cols)

    # save the ts data
    if result_type == 'time_series':
        save_ts_data(ds4, source_dict, db, db_site_id, ds_id, param_name, results_cols, av)
    elif result_type == 'time_series_simulation':
        save_ts_sim_data(ds4, source_dict, db, db_site_id, ds_id, param_name, results_cols, av)


def process_data_H25(ds4, dataset, ds_id, db, source_dict, ds_fields, sites):
    """

    """
    # Get base parameters
    dim_name = 'time'
    param_name = dataset['parameter']
    result_type = ds4[param_name].attrs['result_type']

    # Determine the site parameters from the result parameters
    results_cols = [param_name]

    # Check for extra parameters
    param_attrs = ds4[param_name].attrs.copy()
    if 'ancillary_variables' in param_attrs:
        av = param_attrs['ancillary_variables'].split(' ')
        results_cols.extend(av)
    else:
        av = []

    # Process the dataset fields
    process_dataset_data(db, ds_id, ds4, param_name, ds_fields)

    # If sites are defined, only use those
    if isinstance(sites, list):
        sites_temp = ds4.station_id.values.astype(str)
        ds_sites = sites_temp[np.in1d(sites_temp, sites)]
        # ds_sites = ds4.station_id[ds4.station_id.isin(sites)].values
    else:
        ds_sites = ds4.station_id.values

    for s in ds_sites:

        # Convert xarray H25 format to H23
        ds4a = ds4.loc[{'stationIndex': s, 'station_id': s}]

        if list(ds4a.dims):
            ds5 = ds4a.reset_index('stationIndex', drop=True).swap_dims({'stationIndex': 'time'}).reset_coords('station_id')
        else:
            ds5 = ds4.loc[{'stationIndex': slice(s, s), 'station_id': s}].reset_index('stationIndex', drop=True).swap_dims({'stationIndex': 'time'}).reset_coords('station_id')

        # Process site data
        db_site_id = process_site_data(ds5, db, dim_name, results_cols)

        # save the ts data
        if result_type == 'time_series':
            save_ts_data(ds5, source_dict, db, db_site_id, ds_id, param_name, results_cols, av)
        elif result_type == 'time_series_simulation':
            save_ts_sim_data(ds5, source_dict, db, db_site_id, ds_id, param_name, results_cols, av)


def save_ts_sim_data(ds5, source_dict, db, db_site_id, ds_id, param_name, results_cols, av):
    """

    """
    # Parameters
    ts_coll = db['time_series_simulation']
    site_ds_coll = db['site_dataset']

    encoding = ds5[param_name].encoding.copy()
    scale_factor = encoding['scale_factor']
    precision = int(np.abs(np.log10(scale_factor)))
    mod_date = get_modified_date(ds5.attrs['history'])

    # Process the ts data
    ds_i = ds5[results_cols]
    dict1 = ds_i.to_dict()
    data_array = np.array(dict1['data_vars'][param_name]['data'])
    data_array1 = data_array.round(precision)

    s_json = {'site_id': db_site_id, 'dataset_id': ds_id, 'simulation_date': mod_date, 'modified_date': mod_date, 'from_date': dict1['coords']['time']['data'], 'result': data_array1.tolist()}

    # Check for site_dataset record
    site_ds = site_ds_coll.find_one({'site_id': db_site_id, 'dataset_id': ds_id}, {'_id': 1})

    if site_ds is None:
        # Insert many if first run for better performance
        try:
            ts_coll.insert_one(s_json)
        except BulkWriteError as err:
            print(err.details)
            db_log(db, ds_id, False, db_site_id, err.details)
            raise ValueError(err.details)

        # Insert site_dataset record
        site_ds_dict = {'site_id': db_site_id, 'dataset_id': ds_id}
        sd1 = site_ds_coll.insert_one(site_ds_dict)

        # Get site_dataset id
        site_ds_id = sd1.inserted_id
    else:
        try:
            ts_coll.update_one({'site_id': s_json['site_id'], 'dataset_id': s_json['dataset_id'], 'simulation_date': s_json['simulation_date']}, {'$set': s_json}, upsert=True)
        except BulkWriteError as err:
            print(err.details)
            db_log(db, ds_id, False, db_site_id, err.details)
            raise ValueError(err.details)

        # Get site_dataset id
        site_ds_id = site_ds['_id']

    # Calc the stats
    # min1 = data_array1.min()
    # max1 = data_array1.max()
    # mean1 = data_array1.mean()
    stats_dict = {}

    stats_dict = list(ts_coll.aggregate([{'$match': {'site_id': db_site_id, 'dataset_id': ds_id}}, {'$group': {'_id': None,  'count': {'$sum': 1}, 'from_date': {'$min': '$simulation_date'}, 'to_date': {'$max': '$simulation_date'}}}]))[0]
    stats_dict.pop('_id')

    # Round to the desired precision
    # stats_dict.update({'mean': round(stats_dict['mean'], precision), 'sum': round(stats_dict['sum'], precision), 'min': round(stats_dict['min'], precision), 'max': round(stats_dict['max'], precision)})

    # Insert the stats
    site_ds_coll.update_one({'_id': site_ds_id}, {'$set': {'stats': stats_dict, 'modified_date': mod_date, 'source': source_dict}})


def save_ts_data(ds5, source_dict, db, db_site_id, ds_id, param_name, results_cols, av):
    """

    """
    # Parameters
    ts_coll = db['time_series_result']
    site_ds_coll = db['site_dataset']

    encoding = ds5[param_name].encoding.copy()
    scale_factor = encoding['scale_factor']
    precision = int(np.abs(np.log10(scale_factor)))
    mod_date = get_modified_date(ds5.attrs['history'])

    # Process the ts data
    ds_i = ds5[results_cols]
    s_df = ds_i.to_dataframe().reset_index()
    s_df.rename(columns={param_name: 'result', 'time': 'from_date'}, inplace=True)
    s_df['from_date'] = s_df['from_date'].dt.tz_localize('utc')
    if av:
        s_df['properties'] = s_df[av].to_dict('records')
        s_df.drop(av, axis=1, inplace=True)
    s_df['site_id'] = db_site_id
    s_df['dataset_id'] = ds_id
    mod_date = get_modified_date(ds5.attrs['history'])
    s_df['modified_date'] = mod_date

    # Save the ts data
    s_json = s_df.to_dict('records')

    # Check for site_dataset record
    site_ds = site_ds_coll.find_one({'site_id': db_site_id, 'dataset_id': ds_id}, {'_id': 1})

    if site_ds is None:
        # Insert many if first run for better performance
        try:
            ts_coll.insert_many(s_json)
        except BulkWriteError as err:
            print(err.details)
            db_log(db, ds_id, False, db_site_id, err.details)
            raise ValueError(err.details)

        # Insert site_dataset record
        site_ds_dict = {'site_id': db_site_id, 'dataset_id': ds_id}
        sd1 = site_ds_coll.insert_one(site_ds_dict)

        # Get site_dataset id
        site_ds_id = sd1.inserted_id
    else:
        update_list = []
        for j in s_json:
            update_list.append(UpdateOne({'site_id': j['site_id'], 'dataset_id': j['dataset_id'], 'from_date': j['from_date']}, {'$set': j}, upsert=True))

        try:
            ts_ids = ts_coll.bulk_write(update_list)
        except BulkWriteError as err:
            print(err.details)
            db_log(db, ds_id, False, db_site_id, err.details)
            raise ValueError(err.details)

        # Get site_dataset id
        site_ds_id = site_ds['_id']

    # Calc the stats
    stats_dict = list(ts_coll.aggregate([{'$match': {'site_id': db_site_id, 'dataset_id': ds_id}}, {'$group': {'_id': None,  'min': {'$min': '$result'}, 'max': {'$max': '$result'}, 'mean': {'$avg': '$result'}, 'sum': {'$sum': '$result'}, 'count': {'$sum': 1}, 'from_date': {'$min': '$from_date'}, 'to_date': {'$max': '$from_date'}}}]))[0]
    stats_dict.pop('_id')

    # Round to the desired precision
    stats_dict.update({'mean': round(stats_dict['mean'], precision), 'sum': round(stats_dict['sum'], precision), 'min': round(stats_dict['min'], precision), 'max': round(stats_dict['max'], precision)})

    # Insert the stats
    site_ds_coll.update_one({'_id': site_ds_id}, {'$set': {'stats': stats_dict, 'modified_date': mod_date, 'source': source_dict}})


def db_log(db, dataset_id, success, site_id=None, comments=None, properties=None):
    """

    """
    run_date = pd.Timestamp.now('UTC')

    log_dict = {'date': run_date, 'dataset_id': dataset_id, 'success': success}

    if isinstance(site_id, ObjectId):
        log_dict.update({'site_id': site_id})
    if isinstance(comments, str):
        log_dict.update({'comments': comments})
    if isinstance(properties, dict):
        log_dict.update({'properties': properties})

    log_coll = db['processing_log']
    log_coll.insert_one(log_dict)


def update_db(dataset, sites, from_mod_date):
    """

    """
    # Check to make sure the requested dataset exists in the remote
    prop = get_remote_dataset_properties(dataset)

    if prop is not None:

        # Get the last keys in the db
        client = MongoClient(db_dict['HOST'], password=db_dict['RW_PASSWORD'], username=db_dict['RW_USERNAME'], authSource=db_dict['DATABASE'])
        db = client[db_dict['DATABASE']]

        db_keys, ds_id = get_site_dataset_source(dataset)

        # Get the remote keys after the db keys
        if db_keys.empty:
            remote_keys = get_remote_keys(dataset, from_key_date=from_mod_date)
        else:
            last_key = db_keys.loc[db_keys.LastModified.idxmax(), 'Key']
            remote_keys = get_remote_keys(dataset, start_after=last_key, from_key_date=from_mod_date)

        if not remote_keys.empty:

            ds_fields = get_collection_schema_fields(db, 'dataset')

            # Connection info
            bucket = prop['bucket']
            s3_config = prop['connection_config'].copy()
            s3 = s3_connection(s3_config)

            # Filter sites if required
            if (prop['time_series_type'] == 'H23') and isinstance(sites, list):
                remote_keys_sites = remote_keys.Key.apply(lambda x: x.split('Z/')[1].split('.H23')[0])
                remote_keys = remote_keys[remote_keys_sites.isin(sites)]

            if remote_keys.empty:
                db_log(db, ds_id, True, None, 'No new keys were found')
            else:
                for i, k in remote_keys.iterrows():
                    ds_dict = {}
                    download_objects(k, s3, bucket, ds_dict)
                    ds4 = ds_dict[k.Key]

                    # Make sure station data types are appropriate
                    variables = list(ds4.variables)
                    if 'station_id' in variables:
                        if np.issubdtype(ds4['station_id'].dtype, np.number):
                            attrs1 = ds4['station_id'].attrs
                            ds4['station_id'] = ds4['station_id'].astype(int).astype(str)
                            ds4['station_id'].attrs = attrs1
                    if 'stationIndex' in variables:
                        if np.issubdtype(ds4['stationIndex'].dtype, np.number):
                            attrs1 = ds4['stationIndex'].attrs
                            ds4['stationIndex'] = ds4['stationIndex'].astype(int).astype(str)
                            ds4['stationIndex'].attrs = attrs1

                    # prep source dict
                    source_dict = k.to_dict()
                    source_dict['Size'] = int(source_dict['Size'])

                    # Distribute to appropriate processing function
                    if prop['time_series_type'] == 'H23':
                        process_data_H23(ds4, dataset, ds_id, db, source_dict, ds_fields)
                    elif prop['time_series_type'] == 'H25':
                        process_data_H25(ds4, dataset, ds_id, db, source_dict, ds_fields, sites)

                db_log(db, ds_id, True)
        else:
            db_log(db, ds_id, True, None, 'No new keys were found')


def schedule_datasets(requested_datasets, requested_sites, from_mod_date, scheduling):
    """

    """
    def create_updater(datasets, sites):
        def updater():
            for ds in datasets:
                update_db(ds, sites, from_mod_date)
        return updater

    if ('every' in scheduling) and ('at' in scheduling):

        fun1 = create_updater(requested_datasets, requested_sites)

        if scheduling['every'] == 'hour':
            schedule.every().hour.at(scheduling['at']).do(fun1)

        elif scheduling['every'] == 'day':
            schedule.every().day.at(scheduling['at']).do(fun1)

        elif scheduling['every'] == 'week':
            schedule.every().sunday.at(scheduling['at']).do(fun1)

        else:
            raise ValueError('every must be one of hour, day, or week')

    else:

        freq_times = {'hourly': ':20', 'daily': '00:20', 'weekly': '08:00'}

        freq_ds = {'hourly': [], 'daily': [], 'weekly': []}
        freq_options = list(freq_ds.keys())

        for ds in requested_datasets:
            freq1 = get_remote_dataset_properties(ds)['scheduling']
            if freq1 in freq_options:
                freq_ds[freq1].append(ds)

        freq_ds = {k: v for k, v in freq_ds.items() if v}

        freq_fun = {}
        for f, dss in freq_ds.items():
            freq_fun.update({f: create_updater(dss, requested_sites)})

        # Schedule
        for f, fun in freq_fun.items():
            if f == 'hourly':
                schedule.every().hour.at(freq_times[f]).do(fun)
            elif f == 'daily':
                schedule.every().day.at(freq_times[f]).do(fun)
            elif f == 'weekly':
                schedule.every().sunday.at(freq_times[f]).do(fun)



########################################################
### Clear db

# client = MongoClient(db_dict['HOST'], password=db_dict['RW_PASSWORD'], username=db_dict['RW_USERNAME'], authSource=db_dict['DATABASE'])
# db = client[db_dict['DATABASE']]
#
# loc_coll = db['sampling_site']
# loc_coll.delete_many({})
#
# ts_coll = db['time_series_result']
# ts_coll.delete_many({})
#
# summ_coll = db['site_dataset']
# summ_coll.delete_many({})
#
# ds_coll = db['dataset'].delete_many({})
#
# ds_coll = db['processing_log'].delete_many({})


###################################################################3
### Run main function

# Get the input parameters
requested_datasets, requested_sites, scheduling, from_mod_date, extra_remotes = get_dataset_params(base_dir)

# Set initial delay
sleep(scheduling['delay'])

# Run all jobs initially
for ds in requested_datasets:
    update_db(ds, requested_sites, from_mod_date)

# Schedule jobs
schedule_datasets(requested_datasets, requested_sites, from_mod_date, scheduling)

# Run the scheduler forever!
while True:
    # time1 = pd.Timestamp.now()
    # print(str(time1) + ': Run start')

    schedule.run_pending()
    sleep(1)

    # time2 = pd.Timestamp.now()
    # print(str(time2) + ': Run finish')







################################################################
### Testing



# requested_datasets, requested_sites, scheduling = get_dataset_params()
# dataset = requested_datasets[0]
# sites = requested_sites
#
#
# update_db(dataset, sites)




# sites = ['70105', '70107', '69101']
# sites = ['17603', '11234', '16826']
# # sites = None
# dataset = datasets[0]
# sites = requested_sites
# # datasets = [datasets[1], datasets[4]]
# datasets = [datasets[-2]]
# #
# param = updaters.ecan_hydstra_recorders.main.param.copy()
# param = updaters.niwa_cliflo.main.param.copy()
#
# client = MongoClient(param['db']['host'], password=param['db']['password'], username=param['db']['username'], authSource=param['db']['database'])
# db = client[param['db']['database']]
#
# loc_coll = db['sampling_site']
# loc_coll.delete_many({})
#
# ts_coll = db['time_series_result']
# ts_coll.delete_many({})
#
# summ_coll = db['site_dataset']
# summ_coll.delete_many({})
#
# ds_coll = db['dataset'].delete_many({})
#
# ds_coll = db['processing_log'].delete_many({})
#
# sites1 = get_sites(datasets)
#
# update_site_data(datasets, sites)
# #
# update_ts_data(datasets)
