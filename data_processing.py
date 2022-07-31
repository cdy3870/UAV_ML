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
            
def feature_select(dfs, parsed_names, feats_subset=None):
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

    new_parsed = {}

    sample_key = list(full_parsed.keys())[0]

    for key, value in full_parsed.items():
        if len(full_parsed[key].keys()) == len(full_parsed[sample_key].keys()):
            new_parsed[key] = value

    return new_parsed
                
def get_labels(ids, indexable_meta):
    encode_dict = {"Quadrotor": 0, "Fixed Wing": 1}
    labels = [encode_dict[indexable_meta[id]["type"]] for id in ids]
    return labels


def get_data():
    ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"

    json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
    with open(json_file, 'r') as inputFile:
        meta_json = json.load(inputFile)
    indexable_meta = get_indexable_meta(meta_json)

    with open("ids_matchedfeats.txt", 'rb') as f:
        ids_matchedfeats_dict = pickle.load(f)

    test = []

    # for i in range(5, 9):
    #     ids_file = "new_filtered_ids_" + str(i) + ".txt"

    #     with open(ids_file, 'rb') as f:
    #         test = pickle.load(f)

    #     new_filtered_ids = []
    #     for u in test:
    #         # print(u)
    #         try:
    #             asdf = ids_matchedfeats_dict[u]
    #             new_filtered_ids.append(u)
    #         except:
    #             pass

    #     feats_subset = test[0]

    #     print("Feature Selecting")
    #     full_parsed = {}
    #     count = 0
    #     for u in new_filtered_ids[1:]:
    #         # csv_path = os.path.join("csvFiles", u)
    #         # dfs, names = convert_to_dfs(csv_path)
    #         ulog_path = os.path.join(ulog_folder, u + ".ulg")

    #         dfs, names = convert_to_dfs_ulog(ulog_path)
    #         feature_dict = feature_select(dfs, names, feats_subset=feats_subset)

    #         full_parsed[u] = feature_dict

    #         print("Feature selected: " + str(count) + "/" + str(len(new_filtered_ids)))
    #         count += 1


    #     parsed_file = "full_parsed_" + str(i) + ".txt"
    #     with open(parsed_file, 'wb') as f:
    #         pickle.dump(full_parsed, f)

    with open("full_parsed_7.txt", 'rb') as f:
        full_parsed = pickle.load(f)


    print("Splitting Features")
    full_parsed = split_features(full_parsed)
    new_filtered_ids = list(full_parsed.keys())

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
        print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)))

        count += 1        
        X.append(X_inst) 

    X_data_file = "X_data_7.txt"
    with open(X_data_file, 'wb') as f:
        pickle.dump(X, f)

    with open(X_data_file, 'rb') as f:
        print(len(pickle.load(f)))

    y = get_labels(new_filtered_ids, indexable_meta)

    Y_data_file = "Y_data_7.txt"
    with open(Y_data_file, 'wb') as f:
        pickle.dump(y, f)

    return X, y





def main():
    # json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
    # with open(json_file, 'r') as inputFile:
    #     meta_json = json.load(inputFile)
    # indexable_meta = get_indexable_meta(meta_json)

    # ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
    # ulogs_downloaded = os.listdir(ulog_folder)
    # ulog_ids = [u[:-4] for u in ulogs_downloaded
    #             if indexable_meta[u[:-4]]["type"] == "Quadrotor" or indexable_meta[u[:-4]]["type"] == "Fixed Wing"]

    # print(len(ulog_ids))


    # durations = [duration_to_mill(indexable_meta[u]["duration"]) for u in ulog_ids if indexable_meta[u]["duration"] != "0:00:00"]

    # # print(len(durations))
    # avg_dur = sum(durations)/len(durations)
    # # print(avg_dur)


    # filtered_ids = get_filtered_ids(ulog_ids, indexable_meta, avg_dur)
    # print(len(filtered_ids))



    # grouping_dict = {'id': [], 'type': [], 'airframe': [], 'hardware': [], 'software': [], 'flightModes': [], 'originalTables': [], 'featuresMatched': []}

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


    # print("Feature Matching")
    # new_filtered_ids = []
    # counts_ids_dict = {}
    # ids_feats_dict = {}
    # ids_allfeats_dict = {}

    # desired_feats = get_desired_feats()
    # # print(desired_feats)

    # for u in range(len(filtered_ids)):
    #     ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
    #     names = convert_to_dfs_ulog(ulog_path, only_col_names=True)

    #     n_matched = len(get_n_feats_matched(names))

    #     if n_matched not in counts_ids_dict:
    #         counts_ids_dict[n_matched] = [filtered_ids[u]]
    #     else:
    #         counts_ids_dict[n_matched].append(filtered_ids[u])

    #     ids_feats_dict[filtered_ids[u]] = set(desired_feats).difference(set(names))
    #     ids_allfeats_dict[filtered_ids[u]] = names


    #     if n_matched == 9:
    #         new_filtered_ids.append(filtered_ids[u])

    #     print("Log count: " + str(u) + "/" + str(len(filtered_ids)))


    file_1 = "counts_ids.txt"
    file_2 = "ids_feats.txt"
    file_3 = "ids.txt"
    file_4 = "ids_allfeats.txt"
    file_5 = "ids_matchedfeats.txt"


    # with open(file_1, 'wb') as f:
    #     pickle.dump(counts_ids_dict, f)


    # with open(file_2, 'wb') as f:
    #     pickle.dump(ids_feats_dict, f)


    # with open(file_3, 'wb') as f:
    #     pickle.dump(new_filtered_ids, f)

    # with open(file_4, 'wb') as f:
    #     pickle.dump(ids_allfeats_dict, f)


    # with open(file_5, 'wb') as f:
    #     pickle.dump(ids_matchedfeats_dict, f)






    with open(file_1, 'rb') as f:
        counts_ids_dict = pickle.load(f)


    with open(file_2, 'rb') as f:
        ids_feats_dict = pickle.load(f)

    with open(file_4, 'rb') as f:
        ids_allfeats_dict = pickle.load(f)


    with open(file_5, 'rb') as f:
        ids_matchedfeats_dict = pickle.load(f)

    # print(len(ids_matchedfeats_dict))




    # for x,y in ids_allfeats_dict.items():
    #     print(len(y))

    # matched = frozenset({"test", "test2", "test3"}.intersection({"test", "test2", "test3"}))
    # print(matched)

    same_feats = {}

    sub_list = []

    for i in range(5, 13):
        sub_list += counts_ids_dict[i]


    count = 0
    for i in range(len(sub_list)):
        cand_feats = set(ids_matchedfeats_dict[sub_list[i]])

        for j in range(len(sub_list)):
            # print(comp_feats)

            if i != j:
                comp_feats = ids_matchedfeats_dict[sub_list[j]]
                matched = frozenset(cand_feats.intersection(comp_feats))

                if len(matched) > 0:
                    if matched in same_feats:
                        same_feats[matched].add(sub_list[i])
                    else:
                        same_feats[matched] = set(sub_list[i])  

        print(count)
        count += 1

    # for key, value in same_feats.items():
    #     print(len(key), len(value))


    # for i in range(5, 13):
    #     min_features = i
    #     most_matches = 0
    #     final_feats = frozenset()
    #     for key, value in same_feats.items():
    #         if len(value) > most_matches and len(key) >= min_features:
    #             most_matches = len(value)
    #             final_feats = key

    #     if len(final_feats) != 0:
    #         final_ids = same_feats[final_feats]

    #         print("Num matched feats: " + str(min_features))
    #         print("Total instances: " + str(len(final_ids)))
    #         print("Feature set: ")
    #         print(final_feats)



    # Num matched feats: 5
    # Total instances: 21961
    # Feature set: 
    # frozenset({'vehicle_rates_setpoint', 'battery_status', 'vehicle_attitude_setpoint', 'vehicle_local_position', 'vehicle_gps_position'})
    
    # Num matched feats: 6
    # Total instances: 18623
    # Feature set: 
    # frozenset({'vehicle_rates_setpoint', 'battery_status', 'vehicle_attitude_setpoint', 'vehicle_local_position', 'manual_control_setpoint', 'vehicle_gps_position'})
    
    # Num matched feats: 7
    # Total instances: 14825
    # Feature set: 
    # frozenset({'vehicle_rates_setpoint', 'manual_control_setpoint', 'battery_status', 'vehicle_attitude_setpoint', 'vehicle_local_position', 'home_position', 'vehicle_gps_position'})
    
    # Num matched feats: 8
    # Total instances: 9615
    # Feature set: 
    # frozenset({'vehicle_air_data', 'vehicle_local_position', 'home_position', 'vehicle_gps_position', 'battery_status', 'vehicle_rates_setpoint', 'vehicle_attitude_setpoint', 'manual_control_setpoint'})

    all_ids = []

    for i in range(5, 9):
        min_features = i
        most_matches = 0
        final_feats = frozenset()

        for key, value in same_feats.items():
            if len(value) > most_matches and len(key) == min_features:
                most_matches = len(value)
                final_feats = key

        if len(final_feats) != 0:
            final_ids = list(same_feats[final_feats])
            
        print("Num matched feats: " + str(min_features))
        print("Total instances: " + str(len(final_ids)))
        print("Feature set: ")
        print(final_feats)

        final_ids.insert(0, final_feats)


    # file_name = "new_filtered_ids_" + str(i) + ".txt"

    # with open(file_name, 'wb') as f:
    #     pickle.dump(final_ids, f)

if __name__ == "__main__":
    main()


