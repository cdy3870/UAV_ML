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
from sklearn.preprocessing import StandardScaler
import random
import copy
from imblearn.over_sampling import RandomOverSampler

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

ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"


def update_feature_dict(dfs, parsed_names, feature_dict, table_name, feature_name, cols):
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
            

def look_for_feature(dfs, features):
    flag = False
    found_feature = []
    
    for feature in features:
        for x in list(dfs.keys()):
            for y in list(dfs[x][0].columns):
                if feature == y and x not in found_feature:
                    found_feature.append(x)
            
                
    return found_feature
            
def extract_from_tables(dfs, parsed_names, feats_subset=None):
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




def duration_to_mill(str):
    splitted = str.split(":")
    total_minutes = 0
    
    total_minutes += 3600000*int(splitted[0])
    # print(splitted[1].lstrip("0"))
    
    if splitted[1] != "00":
        total_minutes += 60000*int(splitted[1].lstrip("0"))
    if splitted[2] != "00":
        total_minutes += 1000*int(splitted[2].lstrip("0"))/60
            
    return round(total_minutes, 2)


def get_indexable_meta(meta_json):
    indexable_meta = {}

    for m in meta_json:
        temp_id = m["id"]
        m.pop("id")
        indexable_meta[temp_id] = m
        
    return indexable_meta


def get_filtered_ids(ulog_ids, indexable_meta, avg_dur=None):
    filtered_ids = []

    for u in ulog_ids:
        duration = indexable_meta[u]["duration"]
        if duration != "0:00:00":
            filtered_ids.append(u)

            
    return filtered_ids

def convert_to_dfs_csv(csv_path, only_names=False):
    dfs = {}
    names = []

    # drop ulog file, parameters file, and other files that have configurations essentially (no timestamps)
    # drop_tables = ["ulg", ".DS_Store", "params.txt", "sensor_selection", "parameter_update", "mission_result", "position_setpoint_triplet", "test_motor"]
    drop_tables = []
    
    dirs = os.listdir(csv_path)
    for d in dirs:
        if not any([True if string in d else False for string in drop_tables]):
            temp = d[37:][:-6]
            if temp[len(temp) - 1].isdigit():
                temp = temp[:-2]
                
            # print(temp)
            
            if not only_names:
                if temp not in dfs:
                    dfs[temp] = [pd.read_csv(os.path.join(csv_path, d))]
                else:                    
                    dfs[temp].append(pd.read_csv(os.path.join(csv_path, d)))
                
            names.append(temp)
            
    return dfs, names


def get_desired_feats():
    desired_feats = []
    for i in range(len(feat_list)):
        desired_feats += feat_list[i]["table name"]

    return desired_feats


def get_n_feats_matched(names):
    desired_feats = set(get_desired_feats())

    matched = desired_feats.intersection(set(names))

    return list(matched)
    

def convert_to_dfs_ulog(ulog_path, only_col_names=False, messages=None):
    try:
        log = pyulog.ULog(ulog_path, messages)
    except:
        print("failed to convert " + str(ulog_path) + " to dfs")
        return []

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

    else: return [msg.name for msg in log.data_list]

def replace_nulls(df):
    new_df = df.fillna(0)
    return new_df

def split_features(full_parsed):
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

    new_parsed = {}

    sample_key = list(full_parsed.keys())[0]
    
    for key, value in full_parsed.items():
        if len(full_parsed[key].keys()) == len(full_parsed[sample_key].keys()):
            new_parsed[key] = value

    return new_parsed

def get_labels(ids, indexable_meta, label_name="type"):

    if label_name == "type":
        encode_dict = {"Quadrotor": 0, "Fixed Wing": 1}
        labels = [encode_dict[indexable_meta[id]["type"]] for id in ids]
    elif label_name == "flightMode":
        # all mission flights were manual: Counter({1: 12284})
        only_missions = [id for id in ids if "Mission" in indexable_meta[id]["flightModes"]] 
        labels = [0 if "Manual" in only_missions else 1 for id in ids]

    return labels

def get_mins_maxes(full_parsed):
    mins_maxes = {}
    num_t_ints = 50
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
        # print(ulog_min)
        # print(ulog_max)
        # print("\n")

        mins_maxes[key] = [ulog_min, ulog_max]

    return mins_maxes

def create_intervals(full_parsed, num_t_ints=50):
    print("Creating Intervals")
    intervals = {}
    mins_maxes = get_mins_maxes(full_parsed)
    for key, value in full_parsed.items():
        ulog_min = mins_maxes[key][0]
        ulog_max = mins_maxes[key][1]
        intervals[key] = np.linspace(ulog_min, ulog_max, num_t_ints)
      
    return intervals
                

def timestamp_shorten(full_parsed, keep_percentage):
    # Get the current mins and maxes of each ulog
    mins_maxes = get_mins_maxes(full_parsed)
    full_parsed_copy = copy.deepcopy(full_parsed)
    for key, value in full_parsed_copy.items():
        ulog_min = round(mins_maxes[key][0])
        ulog_max = round(mins_maxes[key][1])
        added_amount = round((ulog_max - ulog_min) * (keep_percentage/100))
        beginning = random.randint(ulog_min, ulog_max - added_amount)
        end = beginning + added_amount

        # print(ulog_min)
        # print(ulog_max)
        # print(beginning)
        # print(end)

        # print(full_parsed_copy[key]['rpy_angles | roll_body'])

        for key_2, value_2 in full_parsed_copy[key].items():
            full_parsed_copy[key][key_2] = [full_parsed_copy[key][key_2][0][full_parsed_copy[key][key_2][0]['timestamp'].between(beginning, end)]]

        # print(full_parsed_copy[key]['rpy_angles | roll_body'])

    return full_parsed_copy

def timestamp_bin(full_parsed, keep_percentage=100, num_t_ints=50):
    print("Timestamp Binning")
    X = []

    if keep_percentage != 100:
        # Shorten the timestamps
        full_parsed = timestamp_shorten(full_parsed, keep_percentage)

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

def feature_select(parse_id="", feats_subset=None, num_tables=7):
    print("Feature Selecting")
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

    if feats_subset == None:
        parsed_file = "full_parsed_" + str(num_tables) + ".txt"
        feats_subset = test[0]

    full_parsed = {}
    count = 0
    for u in new_filtered_ids[1:]:
        ulog_path = os.path.join(ulog_folder, u + ".ulg")

        dfs, names = convert_to_dfs_ulog(ulog_path)
        feature_dict = extract_from_tables(dfs, names, feats_subset=feats_subset)

        full_parsed[u] = feature_dict

        print("Feature selected: " + str(count) + "/" + str(len(new_filtered_ids)))
        count += 1

    parsed_file = "full_parsed_" + str(num_tables) + "_" + parse_id + ".txt"
    with open(parsed_file, 'wb') as f:
        pickle.dump(full_parsed, f)

    return full_parsed


def preprocess_data():
    json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
    with open(json_file, 'r') as inputFile:
        meta_json = json.load(inputFile)
    indexable_meta = get_indexable_meta(meta_json)

    with open("full_parsed_7.txt", 'rb') as f:
        full_parsed = pickle.load(f)


    # full_parsed = feature_select(num_tables=8)
    full_parsed_split = split_features(full_parsed)

    y = get_labels(list(full_parsed_split.keys()), indexable_meta)

    X = timestamp_bin(full_parsed_split)

    X_data_file = "X_data_7.txt"
    with open(X_data_file, 'wb') as f:
        pickle.dump(X, f)


    Y_data_file = "Y_data_7.txt"
    with open(Y_data_file, 'wb') as f:
        pickle.dump(y, f)

    return X, y


def get_stored_data(num_tables, num_t_ints=50, percentage=100):
    if num_t_ints != 50:
        X_data_file = "X_data_" + str(num_tables) + "_" + str(num_t_ints) + "_ints.txt"

    elif percentage != 100:
        X_data_file = "X_data_" + str(num_tables) + "_" + str(percentage) + ".txt"

        with open(X_data_file, 'rb') as f:
            X = pickle.load(f)   
        
        return X      

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

def get_augmented_data(X, y, augment_percent=None):
    X_quad = []
    X_fixed = []
    y_quad = []
    y_fixed = []

    for i in range(len(X)):
        if y[i] == 1:
            y_fixed.append(1)
            X_fixed.append(X[i])
        else:
            y_quad.append(0)
            X_quad.append(X[i])

    # print(len(y_fixed))
        
    if augment_percent == None:
        scale = round(len(X_quad)/len(X_fixed))
    else:
        scale = 1

    my_augmenter = (TimeWarp() * scale + Quantize(n_levels=[10, 20, 30]) + Drift(max_drift=(0.1, 0.5)) @ 0.8 + Reverse() @ 0.5) 

    num_additional_instances = round(len(y_fixed) * augment_percent)
    X_fixed_aug = my_augmenter.augment(np.array(X_fixed)).tolist()[:num_additional_instances]
    y_fixed_aug = [1 for i in range(num_additional_instances)]
    # for inst in num_additional_instances:
    #     y_fixed_aug += [1 for i in range(scale)]

    # print(len(y_fixed_aug))    

    X_aug = X_fixed_aug + X_fixed + X_quad
    y_aug = y_fixed_aug + y_fixed + y_quad

    return X_aug, y_aug

# def shorten_data(X, num_tables, percent):

#     num_t_ints = 50
#     shortened_t_ints = round(num_t_ints * percent)
#     available_t_ints = num_t_ints - shortened_t_ints
#     new_X = np.array(X)

#     for i in range(len(X)):
#         rand_start = random.randint(0, available_t_ints)
#         indices = [i for i in range(rand_start, rand_start + shortened_t_ints)]

#         other_indices = [i for i in range(num_t_ints) if i not in indices]

#         new_X[i, :, other_indices] = 0
#         # print(new_X.shape)

#     return new_X.tolist()

def feature_index(num_tables, indices):
    X, _ = get_stored_data(num_tables)

    new_X = np.array(X)[:, indices, :]

    return new_X.tolist()

def standardize_data(X_train, X_test, independent=False):
    if independent:
        for i in range(X_train.shape[0]):
            X_train_inst = X_train[i]
            scaler = StandardScaler()
            scaler = scaler.fit(X_train_inst)
            X_train_inst = scaler.transform(X_train_inst)
            X_train[i] = X_train_inst

        for i in range(X_test.shape[0]):
            X_test_inst = X_test[i]
            scaler = StandardScaler()
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

        scaler = StandardScaler()
        scaler = scaler.fit(X_train.reshape(num_instances * num_times, num_features))
        X_train = scaler.transform(X_train.reshape(num_instances * num_times, num_features))
        X_test = scaler.transform(X_test.reshape(num_instances_test * num_times_test, num_features_test))

        X_train = X_train.reshape(num_instances, num_features, num_times)
        X_test = X_test.reshape(num_instances_test, num_features_test, num_times_test)

    return X_train, X_test

def random_oversample(X, y, sample_ratio):
    sampler = RandomOverSampler(random_state=0, sampling_strategy=sample_ratio)
    X_np = np.array(X)

    num_instances = X_np.shape[0]
    num_features = X_np.shape[1]
    num_times = X_np.shape[2]


    X_np = X_np.reshape(num_instances, num_features*num_times)
    X_resampled, y_resampled = sampler.fit_resample(X_np, y)

    X_resampled = X_resampled.reshape(X_resampled.shape[0], num_features, num_times).tolist()

    return X_resampled, y_resampled

def main():
    X, y = preprocess_data()


if __name__ == "__main__":
    main()


