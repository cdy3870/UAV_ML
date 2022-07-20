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
            
def feature_select(dfs, parsed_names):
    feature_dict = {}
    
    
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


    for feats in feat_list:  
        update_feature_dict(dfs, parsed_names, feature_dict, feats["table name"], feats["feat(s) name(s)"], feats["feat(s)"])
        # print(feature_dict)
        # try:
        #     update_feature_dict(dfs, parsed_names, feature_dict, feats["table_name"], feats["feat(s) name(s)"], feats["feat(s)"])
        # except:
        #     continue
            # found_feature = look_for_feature(dfs, feats["feat(s)"])
            # print("Looking for: " + str(feats["desc"]))
            # print(found_feature)  

    
    
    
#     # airspeed (not sure)
        
#     # climb rate (not sure)

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


def get_filtered_ids(ulog_ids, indexable_meta, avg_dur):
    filtered_ids = []

    for u in ulog_ids:
        duration = indexable_meta[u]["duration"]
        if duration_to_mill(duration) >= avg_dur or duration != "0:00:00":
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


def convert_to_dfs_ulog(ulog_path, messages=None):
    log = pyulog.ULog(ulog_path, messages)

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

    data = {}
    for msg in log.data_list:
        msg_data = pd.DataFrame.from_dict(msg.data)
        msg_data.columns = [
            col_rename_pattern.sub(
                lambda x: d_col_rename[x.group()], col)
            for col in msg_data.columns
            ]
        # msg_data.index = pd.TimedeltaIndex(msg_data['timestamp'] * 1e3, unit='ns')
        data['{:s}'.format(msg.name)] = [msg_data]
        
    names = [msg.name for msg in log.data_list]
    return data, names

def preprocess(df):
    new_df = df.fillna(0)
    return new_df

def split_features(full_parsed):
    for key, value in full_parsed.items():
        for k in list(full_parsed[key].keys()):
            temp_df = full_parsed[key][k][0]
            # print(temp_df)
            cols = list(temp_df.columns)[1:]
            if len(cols) > 1:
                for col in cols:
                    # print(col)
                    full_parsed[key][k + " | " + col] = [preprocess(temp_df[["timestamp", col]])]
                del full_parsed[key][k]
                

def main():
    json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
    with open(json_file, 'r') as inputFile:
        meta_json = json.load(inputFile)
    indexable_meta = get_indexable_meta(meta_json)

    ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
    ulog_ids = [k for k in list(indexable_meta.keys()) 
                if indexable_meta[k]["type"] == "Quadrotor" or indexable_meta[k]["type"] == "Fixed Wing"]
    # ulog_ids = os.listdir("csvFiles")
    # ulog_ids.remove(".DS_Store")
    print(len(ulog_ids))


    durations = [duration_to_mill(indexable_meta[u]["duration"]) for u in ulog_ids if indexable_meta[u]["duration"] != "0:00:00"]

    # print(len(durations))
    avg_dur = sum(durations)/len(durations)
    # print(avg_dur)


    filtered_ids = get_filtered_ids(ulog_ids, indexable_meta, avg_dur)
    print(len(filtered_ids))



    grouping_dict = {'id': [], 'type': [], 'airframe': [], 'hardware': [], 'software': [], 'flightModes': [], 'originalTables': [], 'featuresMatched': []}

    # print("Grouping")
    # for u in filtered_ids:
    #     # csv_path = os.path.join("csvFiles", u)
    #     ulog_path = os.path.join("dataDownloaded", u + ".ulg")
    #     dfs, names = convert_to_dfs_ulog(ulog_path)
    #     # print([names)
    #     feature_dict = feature_select(dfs, names)    
    #     lenn = len(feature_dict.keys())
        
    #     grouping_dict["id"].append(u)
    #     grouping_dict["type"].append(indexable_meta[u]["type"])
    #     grouping_dict["airframe"].append(indexable_meta[u]["airframe"])
    #     grouping_dict["hardware"].append(indexable_meta[u]["hardware"])
    #     grouping_dict["software"].append(indexable_meta[u]["software"])
    #     grouping_dict["flightModes"].append(indexable_meta[u]["flightModes"])
    #     grouping_dict["originalTables"].append(names)
    #     grouping_dict["featuresMatched"].append(lenn)




    # grouping_df = pd.DataFrame(grouping_dict)
    # grouped_by_counts = grouping_df.sort_values(by=['featuresMatched']).reset_index(drop=True)


    print("Feature Selecting")
    full_parsed = {}
    # new_filtered_ids = list(grouping_df[grouping_df["featuresMatched"] == 11]["id"])
    # print(new_filtered_ids)
    for u in filtered_ids:
        # csv_path = os.path.join("csvFiles", u)
        # dfs, names = convert_to_dfs(csv_path)
        ulog_path = os.path.join(ulog_folder, u + ".ulg")
        dfs, names = convert_to_dfs_ulog(ulog_path)
        feature_dict = feature_select(dfs, names)

        if len(feature_dict.keys()) == 11:
            full_parsed[u] = feature_dict


    print("Splitting Features")
    split_features(full_parsed)

    print("Creating Intervals")
    intervals = {}
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
        intervals[key] = np.linspace(ulog_min, ulog_max, num_t_ints)
        
            # print(full_parsed[key][key_2][0].shape[0])


    X = []

    print(len(full_parsed))

    print("Timestamp Binning")
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
        
        X.append(X_inst) 

    X_data_file = "X_data.txt"
    with open(X_data_file, 'wb') as f:
        pickle.dump(X, f)

    with open(X_data_file, 'rb') as f:
        print(len(pickle.load(f)))

    # print(X)

if __name__ == "__main__":
    main()


