import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyulog as plog
import re
import bisect
import math
import pyulog
import pickle
import json
from collections import Counter
from itertools import combinations, islice
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import copy
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, ClusterCentroids

work_dir = "../../../work/uav-ml/"
ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
ulog_folder_hex = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloadedHex"


################################################## FEATURE EXTRACTION/SELECTION ########################################################

feat_list = [
            {"desc": "pos | local position", "table name": ["vehicle_local_position"], "feat(s) name(s)": "loc_pos", "feat(s)": ["x", "y", "z"]},
    
            {"desc": "pos | ground speed", "table name": ["sensor_gps", "vehicle_gps_position"], "feat(s) name(s)": "ground_speed", "feat(s)": ["vel_n_m_s", "vel_e_m_s", "vel_d_m_s"]},
    
            {"desc": "pos | roll, pitch, yaw angles", "table name": ["vehicle_attitude_setpoint"], "feat(s) name(s)": "rpy_angles", "feat(s)": ["roll_body", "pitch_body", "yaw_body"]},
    
            {"desc": "pos | roll, pitch, yaw speeds ", "table name": ["vehicle_rates_setpoint", "rollspeed"], "feat(s) name(s)": "rpy_speeds", "feat(s)": ["roll", "pitch", "yaw"]},
    
            {"desc": "pos | relative altitude", "table name": ["home_position"], "feat(s) name(s)": "rel_alt", "feat(s)": ["alt"]},
    
            {"desc": "pos | local altitude", "table name": ["sensor_gps", "vehicle_gps_position"], "feat(s) name(s)": "loc_alt", "feat(s)": ["alt"]},
    
            {"desc": "pos | quaternion", "table name": ["vehicle_attitude_setpoint"], "feat(s) name(s)": "quat", "feat(s)": ["q_d[0]", "q_d[1]", "q_d[2]", "q_d[3]"]},
    
            {"desc": "imu | acceleration", "table name": ["sensor_accel"], "feat(s) name(s)": "accel", "feat(s)": ["x", "y", "z"]},
    
            {"desc": "imu | angular speed", "table name": ["sensor_gyro"], "feat(s) name(s)": "ang_speed", "feat(s)": ["x", "y", "z"]},
    
            {"desc": "imu | magnetic field", "table name": ["sensor_mag"], "feat(s) name(s)": "mag_field", "feat(s)": ["x", "y", "z"]},
    
            {"desc": "imu | absolute pressure", "table name": ["vehicle_air_data"], "feat(s) name(s)": "abs_pressure", "feat(s)": ["baro_pressure_pa"]},
    
            {"desc": "imu | pressure altitude", "table name": ["vehicle_air_data"], "feat(s) name(s)": "pressure_alt", "feat(s)": ["baro_alt_meter"]},
    
            {"desc": "sys | battery temperature", "table name": ["battery_status"], "feat(s) name(s)": "batt_temp", "feat(s)": ["temperature"]},
    
            {"desc": "sys | heading", "table name": ["vehicle_local_position"], "feat(s) name(s)": "heading", "feat(s)": ["heading"]},
    
            {"desc": "sys | throttle", "table name": ["manual_control_setpoint"], "feat(s) name(s)": "throttle", "feat(s)": ["z"]}]


def get_indexable_meta(meta_json):
    '''
    Get mapped ulog ids to its metadata (duration, drone type, etc.)

    Parameters:
        meta_json (dict) : mapped data

    Returns:
        indexable_meta (dict) :  mapped metadata
    '''

    indexable_meta = {}

    for m in meta_json:
        temp_id = m["id"]
        m.pop("id")
        indexable_meta[temp_id] = m
        
    return indexable_meta

json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
with open(json_file, 'r') as inputFile:
    meta_json = json.load(inputFile)
indexable_meta = get_indexable_meta(meta_json)

def update_feature_dict(dfs, parsed_names, feature_dict, table_name, feature_name, cols):
    '''

    Parameters:
        dfs
        parsed_names
        feature_dict
        table_name
        feature_name
        cols
    '''

    spec_df = []
    if len(table_name) > 1:
        for t in table_name:
            try:
                spec_df = dfs[table_name] 
            except:
                pass
        if len(spec_df) == 0:
            return
    else:
        # table not found
        if table_name[0] not in dfs.keys():
            return

        spec_df = dfs[table_name[0]] 
        
        # if specific feature not found
        if cols[0] not in list(spec_df[0].columns):
            return

    # for d in spec_df:
    if table_name[0] not in feature_dict:
        feature_dict[feature_name] = [spec_df[0][["timestamp"] + cols]]
    else:
        feature_dict[feature_name].append(spec_df[0][["timestamp"] + cols])
            

def extract_individual(dfs, feats_subset):
    '''
    Returns chunk of data between two timestamps

    Parameters:
        dfs () : 
        feats_subset () :
        
    Returns:
        feature_dict () :
    '''

    feature_dict = {}
    for feat in feats_subset:
        # print(feat)
        strings = feat.split(" ")
        table_name = strings[0]
        feat_name = strings[2]

        feature_dict[feat] = [dfs[table_name][0][["timestamp"] + [feat_name]]]


    return feature_dict
     

def extract_from_tables(dfs, parsed_names, feats_subset=None):
    '''

    Parameters:
        dfs () :
        parsed_names () :
        feats_subset () :
        
    Returns:
        feature_dict () :
    '''

    feature_dict = {}

    if feats_subset == None:
        for feats in feat_list:  
            update_feature_dict(dfs, parsed_names, feature_dict, feats["table name"], feats["feat(s) name(s)"], feats["feat(s)"])
    else:
        for feats in feat_list:  
            for f in feats["table name"]:
                if f in feats_subset:
                    update_feature_dict(dfs, parsed_names, feature_dict, feats["table name"], feats["feat(s) name(s)"], feats["feat(s)"])
                    break


    return feature_dict


def get_distribution(ids):
    '''
    Returns the distribution of classes for the provided ids

    Parameters:
        ids (list) : list of ulog ids

    Returns:
        distribution (dict) : distribution of drone types
    '''

    types = []

    for u in ids:
        types.append(indexable_meta[u]["type"])

    return Counter(types)


def get_filtered_ids():
    '''
    Returns the ids from the quad, fixed, and hex folders

    Returns:
        filtered_ids (list) : ids used in experiments
    '''

    ulogs_downloaded = os.listdir(ulog_folder)
    ulogs_downloaded_hex = os.listdir(ulog_folder_hex)


    ulogs_downloaded = ulogs_downloaded_hex + ulogs_downloaded 


    drone_ids = [u[:-4] for u in ulogs_downloaded]

    filtered_ids = [u for u in drone_ids if indexable_meta[u]["duration"] != "0:00:00"]

            
    return filtered_ids


def convert_to_dfs_ulog(ulog_path, specific_tables=[], only_col_names=False):
    '''
    Parses ulog file and extracts data as dataframes

    Parameters:
        ulog_path (string) : path to ulog file
        specific_tables (list) : list of topic or table names for ulogs
        only_col_names (bool) : only return the topics of a ulog and not its data

    Returns:
        dataframes () :
    '''

    try:
        log = pyulog.ULog(ulog_path, messages=None)
    except:
        print("failed to convert " + str(ulog_path) + " to dfs")
        return {}, []

    # column naming
    d_col_rename = {
        '[': '_',
        ']': '_',
        '.': '_',
    }
    col_rename_pattern = re.compile(
        r'(' + '|'.join([
                            re.escape(key)
                            for key in d_col_rename.keys()]) + r')')


    if not only_col_names:
        data = {}
        names = []

        if len(specific_tables) == 0:
            for msg in log.data_list:
                msg_data = pd.DataFrame.from_dict(msg.data)
                names.append(msg.name)
                msg_data.columns = [
                    col_rename_pattern.sub(
                        lambda x: d_col_rename[x.group()], col)
                    for col in msg_data.columns
                    ]
                # msg_data.index = pd.TimedeltaIndex(msg_data['timestamp'] * 1e3, unit='ns')
                data['{:s}'.format(msg.name)] = [msg_data]

            return data, names
        else:
            for msg in log.data_list:
                if msg.name in specific_tables:
                    msg_data = pd.DataFrame.from_dict(msg.data)
                    names.append(msg.name)
                    msg_data.columns = [
                        col_rename_pattern.sub(
                            lambda x: d_col_rename[x.group()], col)
                        for col in msg_data.columns
                        ]
                    # msg_data.index = pd.TimedeltaIndex(msg_data['timestamp'] * 1e3, unit='ns')
                    data['{:s}'.format(msg.name)] = [msg_data]       

        return data, names     

    else: return [msg.name for msg in log.data_list]

def replace_nulls(df):
    '''
    Replace null values with 0s

    Parameters:
        df (pd.DataFrame) : data dataframe
        
    Returns:
        new_df (pd.DataFrame) : dataframe with no nulls
    '''

    new_df = df.fillna(0)
    return new_df

def split_features(full_parsed):
    '''
    Splits mapped data into individual feature names for each ulog and
    ensures that each parsing has the same number of features

    Parameters:
        full_parsed (dict) : mapped data
        
    Returns:
        new_parsed (dict) : split data
    '''

    print("Splitting Features")
    for key, value in full_parsed.items():
        for k in list(full_parsed[key].keys()):
            temp_df = full_parsed[key][k][0]
            # print(temp_df)
            cols = list(temp_df.columns)[1:]
            if len(cols) > 1:
                for col in cols:
                    # print(col)
                    full_parsed[key][k + " | " + col] = [replace_nulls(temp_df[["timestamp", col]])]
                del full_parsed[key][k]


    sample_key = list(full_parsed.keys())[0]

    keys_dict = {}
    
    for key, value in full_parsed.items():
        keys = set(full_parsed[key].keys())

        if frozenset(keys) in keys_dict:
            keys_dict[frozenset(keys)] += 1
        else:
            keys_dict[frozenset(keys)] = 1

    most_features = max(keys_dict, key=keys_dict.get)


    new_parsed = {}
    for key, value in full_parsed.items():
        if full_parsed[key].keys() == set(most_features):
            new_parsed[key] = value


    return new_parsed

def get_labels(ids):
    '''
    Returns the labels (drone type) for each ulog given the ids

    Parameters:
        ids (list) : list of ulog ids

    Returns:
        labels (list) : list of drone types
    '''

    encode_dict = {"Quadrotor": 0, "Fixed Wing": 1, "Hexarotor": 2}
    labels = [encode_dict[indexable_meta[id]["type"]] for id in ids]

    return labels

def get_mins_maxes(full_parsed):
    '''
    Returns the minimum and maximum timestamp for each flights

    Parameters:
        full_parsed (dict) : mapped data
        
    Returns:
        mins_maxes (dict) : the minimum and maximum timestamp for each flight
    '''

    mins_maxes = {}


    for key, value in full_parsed.items():
        ulog_max = 0
        ulog_min = float('inf')
        for key_2, value_2 in full_parsed[key].items():
            min = full_parsed[key][key_2][0]["timestamp"].min()
            max = full_parsed[key][key_2][0]["timestamp"].max()
            
            if min < ulog_min:
                ulog_min = min
            if max > ulog_max:
                ulog_max = max

        mins_maxes[key] = [ulog_min, ulog_max]

    return mins_maxes

def create_intervals(full_parsed, num_t_ints=50):
    '''
    Finds the timestamp boundaries when dividing a up a flight into a certain
    number of intervals

    Parameters:
        full_parsed (dict) : mapped data
        num_t_ints (int) : number of intervals
        
    Returns:
        intervals (dict) : the timestamp boundaries for each divided flight
    '''
    
    intervals = {}
    mins_maxes = get_mins_maxes(full_parsed)
    for key, value in full_parsed.items():
        ulog_min = mins_maxes[key][0]
        ulog_max = mins_maxes[key][1]
        intervals[key] = np.linspace(ulog_min, ulog_max, num_t_ints)
      
    return intervals
                

def timestamp_bin(full_parsed, keep_percentage=100, num_t_ints=50, beg_mid_end=None):
    '''
    Sampling technique that divides flight into equal width intervals and averages
    values

    Parameters:
        full_parsed (dict) : mapping of flight ids to data
        keep_percentage (int) : percentage of flight that is kept
        num_t_ints (int) : number of intervals
        beg_mid_end (string) : if we remove from the beginning, middle, or end of flight

    Returns:
        X (list) : timestamp binned data
    '''
    

    print("Timestamp Binning")
    X = []

    if keep_percentage != 100:
        # Shorten the timestamps
        full_parsed = timestamp_shorten(full_parsed, keep_percentage, beg_mid_end)

        # Get the new intervals
        intervals = create_intervals(full_parsed, num_t_ints)
    else:
        intervals = create_intervals(full_parsed, num_t_ints)


    count = 0
    for key, value in full_parsed.items():
        X_inst = [[0 for i in range(num_t_ints)] for i in range(len(full_parsed[key]))]
        index = 0
        # example = full_parsed[id]
        # print(key)

        temp_dict = {}
        for key_2, value_2 in full_parsed[key].items():
            for i in range(full_parsed[key][key_2][0].shape[0]):
                interval_index = bisect.bisect_left(intervals[key], full_parsed[key][key_2][0].iloc[i]["timestamp"])
                # print(interval_index)
                if i == 0:
                    prev_interval = interval_index
                    beg_index = 0

                if interval_index != prev_interval:
                    temp_dict[prev_interval] = (beg_index, i - 1)
                    beg_index = i
                    prev_interval = interval_index


            for i in range(num_t_ints):
                if i in temp_dict:
                    left = temp_dict[i][0]
                    right = temp_dict[i][1]

                    avg = list(full_parsed[key][key_2][0].iloc[left:right, 1:].mean())[0]

                    if math.isnan(avg):
                        X_inst[index][i] = 0
                    else:
                        X_inst[index][i] = avg

            index += 1




            # print(temp_dict)
            temp_dict = {}
        print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)) + ", Keep Percentage: " + str(keep_percentage))

        count += 1        
        X.append(X_inst) 

    # print(np.array(X).shape)

    return X

def get_desired_distribution(full_parsed_split, ids, total_quads=None, total_fixed=None, total_hex=None):
    distribution = get_distribution(ids)

    new_parsed = {}
    new_y = []

    if total_quads == None and total_fixed == None and total_hex == None:  
        total_quads = 1000
        total_fixed = distribution["Fixed Wing"]
        total_hex = distribution["Hexarotor"]
    else:
        total_quads = min(total_quads, distribution["Quadrotor"])
        total_fixed = min(total_fixed, distribution["Fixed Wing"])
        total_hex = min(total_hex, distribution["Hexarotor"])


    quad_count = 0
    fixed_count = 0
    hex_count = 0
    keys = []

    for key in full_parsed_split.keys():
        type = indexable_meta[key]["type"]
        if type == "Quadrotor":
            if quad_count < total_quads:
                new_parsed[key] = full_parsed_split[key]
                new_y.append(0)
                keys.append(key)    
            quad_count += 1

        elif type == "Fixed Wing":
            if fixed_count < total_fixed:
                new_parsed[key] = full_parsed_split[key]
                new_y.append(1)
                keys.append(key)    
            fixed_count += 1

        else:
            if hex_count < total_hex:
                new_parsed[key] = full_parsed_split[key]
                new_y.append(2)
                keys.append(key)    
            hex_count += 1      



        if quad_count == total_quads and fixed_count == total_fixed and hex_count == total_hex:
            break

    return new_parsed, keys

def feature_select(parse_id="", feats_subset=None):
    filtered_ids = get_filtered_ids()
    full_parsed = {}
    count = 0


    _, keys = get_desired_distribution({key:{} for key in filtered_ids}, filtered_ids, 1000, 500, 500)

    print(len(keys))

    for u in keys:
        if indexable_meta[u]["type"] == "Hexarotor":
            ulog_path = os.path.join(ulog_folder_hex, u + ".ulg")
        else:
            ulog_path = os.path.join(ulog_folder, u + ".ulg")

        dfs, names = convert_to_dfs_ulog(ulog_path)


        try:
            feature_dict = extract_individual(dfs, feats_subset=feats_subset)
            full_parsed[u] = feature_dict

            print("Feature selected: " + str(count))
            distribution = get_distribution(list(full_parsed.keys()))
            print(distribution)

            count += 1
        except:
            continue



    parsed_file = "full_parsed_" + parse_id + ".txt"
    with open(parsed_file, 'wb') as f:
        pickle.dump(full_parsed, f)

    return full_parsed

def feature_select_from_paper(parse_id="", feats_subset=None, num_tables=7):
    print("Feature Selecting")


    if feats_subset == None:
        with open("ids_matchedfeats.txt", 'rb') as f:
            ids_matchedfeats_dict = pickle.load(f)

        test = []


        ids_file = "new_filtered_ids_" + str(num_tables) + ".txt"

        with open(ids_file, 'rb') as f:
            test = pickle.load(f)

        new_filtered_ids = []
        for u in test:
            # print(u)
            try:
                asdf = ids_matchedfeats_dict[u]
                new_filtered_ids.append(u)
            except:
                pass

            parsed_file = "full_parsed_" + str(num_tables) + ".txt"
            feats_subset = test[0]

    print(feats_subset)


    filtered_ids = get_filtered_ids()

    full_parsed = {}
    count = 0
    for u in filtered_ids:
        if indexable_meta[u]["type"] == "Hexarotor":
            ulog_path = os.path.join(ulog_folder_hex, u + ".ulg")
        else:
            ulog_path = os.path.join(ulog_folder, u + ".ulg")

        dfs, names = convert_to_dfs_ulog(ulog_path)

        if len(set(names).intersection(feats_subset)) == len(feats_subset):
            feature_dict = extract_from_tables(dfs, names, feats_subset=feats_subset)

            full_parsed[u] = feature_dict
            # print(feats_subset)
            # print(names)
            # print(feature_dict.keys())

        print("Feature selected: " + str(count) + "/" + str(len(filtered_ids)))
        count += 1

    parsed_file = "full_parsed_" + str(num_tables) + "_" + parse_id + ".txt"
    with open(parsed_file, 'wb') as f:
        pickle.dump(full_parsed, f)

    return full_parsed


def preprocess_data(X_file, y_file, saved_parse=None, parse_id="", feats_subset=None):
    if saved_parse != None:
        with open(saved_parse, 'rb') as f:
            full_parsed = pickle.load(f)
    else:
        full_parsed = feature_select(parse_id=parse_id, feats_subset=feats_subset)

    # full_parsed_split = split_features(full_parsed)

    y = get_labels(list(full_parsed.keys()))

    # print(Counter(y))
    # full_parsed = dict(list(full_parsed.items())[:10]) 

    X = timestamp_bin(full_parsed)


    with open(X_file, 'wb') as f:
        pickle.dump(X, f)


    with open(y_file, 'wb') as f:
        pickle.dump(y, f)


    return X, y



def get_stored_data(num_tables, num_t_ints=50, percentage=100, beg_mid_end="", X_path="", Y_path=""):
    if num_t_ints != 50:
        X_data_file = "X_data_" + str(num_tables) + "_" + str(num_t_ints) + "_ints.txt"

    elif percentage != 100:
        X_data_file = "X_data_" + str(num_tables) + "_" + str(percentage) + ".txt"

        with open(X_data_file, 'rb') as f:
            X = pickle.load(f)   
        
        return X  
    elif beg_mid_end != "":
        X_data_file = "X_data_" + str(num_tables) + "_" + beg_mid_end + ".txt"

        with open(X_data_file, 'rb') as f:
            X = pickle.load(f)   
        
        return X  
    elif X_path != "" and Y_path != "":
        with open(X_path, 'rb') as f:
            X = pickle.load(f)   

        with open(Y_path, 'rb') as f:
            y = pickle.load(f)   

        return X, y
    else:
        X_data_file = "formatted_data/X_data_" + str(num_tables) + ".txt"



    Y_data_file = "formatted_data/Y_data_" + str(num_tables) + ".txt"


    with open(X_data_file, 'rb') as f:
        X = pickle.load(f) 

    with open(Y_data_file, 'rb') as f:
        y = pickle.load(f) 

    # Counter({0: 9255, 1: 359})
    # print(Counter(y))

    # one broken instance
    # X.pop(2898)
    # y.pop(2898)

    return X, y

################################################## INPUT DATA MODIFICATION ########################################################
def timestamp_shorten(full_parsed, keep_percentage, beg_mid_end):
    mins_maxes = get_mins_maxes(full_parsed)
    full_parsed_copy = copy.deepcopy(full_parsed)
    for key, value in full_parsed_copy.items():
        ulog_min = round(mins_maxes[key][0])
        ulog_max = round(mins_maxes[key][1])

        if beg_mid_end == None:
            added_amount = round((ulog_max - ulog_min) * (keep_percentage/100))
            start = random.randint(ulog_min, ulog_max - added_amount)
            end = start + added_amount
        else:
            third = round((ulog_max - ulog_min) * (0.33))
            if beg_mid_end == "beg":
                start = ulog_min
                end = start + third
            elif beg_mid_end == "mid":
                start = ulog_min + third
                end = start + third
            elif beg_mid_end == "end":
                start = ulog_min + 2*third
                end = ulog_max               

        for key_2, value_2 in full_parsed_copy[key].items():
            full_parsed_copy[key][key_2] = [full_parsed_copy[key][key_2][0][full_parsed_copy[key][key_2][0]['timestamp'].between(start, end)]]

    return full_parsed_copy

def get_augmented_data(X, y, augment_percent=None):
    X_quad = []
    X_fixed = []
    y_quad = []
    y_fixed = []
    X_hex = []
    y_hex = []

    for i in range(len(X)):
        if y[i] == 1:
            y_fixed.append(1)
            X_fixed.append(X[i])
        elif y[i] == 0:
            y_quad.append(0)
            X_quad.append(X[i])
        else:
            y_hex.append(2)
            X_hex.append(X[i])


    # print(len(y_fixed))
    # print(len(y_hex))

    if augment_percent == None:
        scale = round(len(X_quad)/len(X_fixed))
        my_augmenter = (TimeWarp() * scale + Quantize(n_levels=[10, 20, 30]) + Drift(max_drift=(0.1, 0.5)) @ 0.8 + Reverse() @ 0.5) 
        X_fixed_aug = my_augmenter.augment(np.array(X_fixed)).tolist()
        y_fixed_aug = [1 for i in range(len(X_fixed_aug))]
    else:
        scale = 5
        my_augmenter = (TimeWarp() * scale + Quantize(n_levels=[10, 20, 30]) + Drift(max_drift=(0.1, 0.5)) @ 0.8 + Reverse() @ 0.5) 
        num_additional_instances = round(len(y_fixed) * augment_percent)
        X_fixed_aug = my_augmenter.augment(np.array(X_fixed)).tolist()[:num_additional_instances]
        y_fixed_aug = [1 for i in range(num_additional_instances)]

        num_additional_instances = round(len(y_hex) * augment_percent)
        X_hex_aug = my_augmenter.augment(np.array(X_hex)).tolist()[:num_additional_instances]
        y_hex_aug = [2 for i in range(num_additional_instances)]

    X_aug = X_fixed_aug + X_hex_aug + X_fixed + X_quad + X_hex
    y_aug = y_fixed_aug + y_hex_aug + y_fixed + y_quad + y_hex

    # print(len(X_fixed_aug))
    # print(len(X_hex_aug))

    return X_aug, y_aug

def feature_index(num_tables, indices):
    '''
    Applies normalization to train and test data

    Parameters:
        X_train (np.array) : train data from sklearn's train_test_split() function
        X_test (np.array) : test data from sklearn's train_test_split() function
        scaler_type (string) : sklearn's standard or min max scaler
        independent (bool) : if data standardizes based on a global standard deviation and mean
        rather than a local one based on current flight

     Returns:
        X_train (np.array) : modified train data
        X_test(np.array) : modified test data   
    '''

    X, _ = get_stored_data(num_tables)

    new_X = np.array(X)[:, indices, :]

    return new_X.tolist()

def standardize_data(X_train, X_test, scaler_type="standard", independent=False):
    '''
    Applies normalization to train and test data

    Parameters:
        X_train (np.array) : train data from sklearn's train_test_split() function
        X_test (np.array) : test data from sklearn's train_test_split() function
        scaler_type (string) : sklearn's standard or min max scaler
        independent (bool) : if data standardizes based on a global standard deviation and mean
        rather than a local one based on current flight

     Returns:
        X_train (np.array) : modified train data
        X_test(np.array) : modified test data   
    '''

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "min_max":
        scaler = MinMaxScaler()

    if independent:
        for i in range(X_train.shape[0]):
            X_train_inst = X_train[i]
            scaler = scaler.fit(X_train_inst)
            X_train_inst = scaler.transform(X_train_inst)
            X_train[i] = X_train_inst

        for i in range(X_test.shape[0]):
            X_test_inst = X_test[i]
            scaler = scaler.fit(X_test_inst)
            X_test_inst = scaler.transform(X_test_inst)
            X_test[i] = X_test_inst

    else:
        num_instances = X_train.shape[0]
        num_features = X_train.shape[1]
        num_times = X_train.shape[2]
        num_instances_test = X_test.shape[0]
        num_features_test = X_test.shape[1]
        num_times_test = X_test.shape[2]

        scaler = scaler.fit(X_train.reshape(num_instances * num_times, num_features))
        X_train = scaler.transform(X_train.reshape(num_instances * num_times, num_features))
        X_test = scaler.transform(X_test.reshape(num_instances_test * num_times_test, num_features_test))

        X_train = X_train.reshape(num_instances, num_features, num_times)
        X_test = X_test.reshape(num_instances_test, num_features_test, num_times_test)

    return X_train, X_test

def apply_sampling(X, y, sample_method, sample_ratio):
    '''
    Applies oversampling and undersampling technique to provided data

    Parameters:
        X (list) : flight data as lists
        y (list) : flight labels
        sample_method (string) : the sampling method
        sample_ratio (int) : percentage to undersample or oversample

    Returns:
        X_resampled (np.array) : modified data
        y_resampled (np.array) : modified data
    '''

    ratio_dict = {}
    distribution = Counter(y)
    for i in range(len(sample_ratio)):
        ratio_dict[i] = int(distribution[i] * (sample_ratio[i]/100))

    if sample_method == "ros":
        sampler = RandomOverSampler(random_state=0, sampling_strategy=ratio_dict)
    elif sample_method == "rus":
        sampler = RandomUnderSampler(random_state=0, sampling_strategy=ratio_dict)
    elif sample_method == "nn":
        sampler = ClusterCentroids(random_state=0, sampling_strategy=ratio_dict)
    elif sample_method == "smote":
        sampler = SMOTE(random_state=0, sampling_strategy=ratio_dict)

    X_np = np.array(X)

    num_instances = X_np.shape[0]
    num_features = X_np.shape[1]
    num_times = X_np.shape[2]


    X_np = X_np.reshape(num_instances, num_features*num_times)
    X_resampled, y_resampled = sampler.fit_resample(X_np, y)

    X_resampled = X_resampled.reshape(X_resampled.shape[0], num_features, num_times).tolist()

    # print(X_resampled.shape)

    return X_resampled, y_resampled



def main():
    X, y = preprocess_data()


if __name__ == "__main__":
    main()



# Deprecated functions

# def look_for_feature(dfs, features):
#     flag = False
#     found_feature = []
    
#     for feature in features:
#         for x in list(dfs.keys()):
#             for y in list(dfs[x][0].columns):
#                 if feature == y and x not in found_feature:
#                     found_feature.append(x)
            
                
#     return found_feature

# def get_desired_feats():
#     '''
        
#     Returns:
#         desired_feats () :
#     '''

#     desired_feats = []
#     for i in range(len(feat_list)):
#         desired_feats += feat_list[i]["table name"]

#     return desired_feats


# def get_n_feats_matched(names):
#     '''

#     Parameters:
#         names () :
        
#     Returns:
#         matched_feats () :
#     '''

#     desired_feats = set(get_desired_feats())

#     matched = desired_feats.intersection(set(names))

#     return list(matched)




