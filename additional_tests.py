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
import data_processing as dp
import _pickle as cpickle
import time
import gc
from multiprocessing import Pool
import os.path
import random

baseline_feats = ["vehicle_local_position | x", "vehicle_local_position | y",
				 "vehicle_local_position | z", "vehicle_attitude_setpoint | roll_body",
				 "vehicle_attitude_setpoint | pitch_body", "vehicle_attitude_setpoint | yaw_body"]

current_feats = ["vehicle_local_position | x", "vehicle_local_position | y",
				 "vehicle_local_position | z", "vehicle_attitude_setpoint | roll_body",
				 "vehicle_attitude_setpoint | pitch_body", "vehicle_attitude_setpoint | yaw_body",
				 "manual_control_setpoint | z", "sensor_gps | alt", "battery_status | temperature"]


ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
ulog_folder_hex = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloadedHex"
csv_folder = "../../../work/uav-ml/px4-Ulog-Parsers/csvFiles" 
json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
work_dir = "../../../work/uav-ml/"

with open("full_parsed_7_multi.txt", "rb") as f:
	full_parsed_split = dp.split_features(pickle.load(f))
	temp_ids = list(full_parsed_split.keys())
temp_mapping = {}


json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
with open(json_file, 'r') as inputFile:
	meta_json = json.load(inputFile)
indexable_meta = dp.get_indexable_meta(meta_json)

with open(work_dir + "ids_allfeats.txt", 'rb') as f:
	ids_allfeats = pickle.load(f)

with open(work_dir + "ids_alltables.txt", 'rb') as f:
	ids_allfeats_dict = pickle.load(f)

def feature_matching_test():
	print("test")
	json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
	with open(json_file, 'r') as inputFile:
		meta_json = json.load(inputFile)
	types = []
	for i in meta_json:
		types.append(i["type"])
	print(Counter(types))

	print("Feature Matching")
	new_filtered_ids = []
	counts_ids_dict = {}
	ids_feats_dict = {}
	ids_allfeats_dict = {}

	desired_feats = get_desired_feats()
	# print(desired_feats)

	for u in range(len(filtered_ids)):
		ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
		names = convert_to_dfs_ulog(ulog_path, only_col_names=True)

		n_matched = len(get_n_feats_matched(names))

		if n_matched not in counts_ids_dict:
			counts_ids_dict[n_matched] = [filtered_ids[u]]
		else:
			counts_ids_dict[n_matched].append(filtered_ids[u])

		ids_feats_dict[filtered_ids[u]] = set(desired_feats).difference(set(names))
		ids_allfeats_dict[filtered_ids[u]] = names


		if n_matched == 9:
			new_filtered_ids.append(filtered_ids[u])

		print("Log count: " + str(u) + "/" + str(len(filtered_ids)))




			


	file_1 = "counts_ids.txt"
	file_2 = "ids_feats.txt"
	file_3 = "ids.txt"
	file_4 = "ids_allfeats.txt"
	file_5 = "ids_matchedfeats.txt"


	with open(file_1, 'wb') as f:
		pickle.dump(counts_ids_dict, f)


	with open(file_2, 'wb') as f:
		pickle.dump(ids_feats_dict, f)


	with open(file_3, 'wb') as f:
		pickle.dump(new_filtered_ids, f)

	with open(file_4, 'wb') as f:
		pickle.dump(ids_allfeats_dict, f)


	with open(file_5, 'wb') as f:
		pickle.dump(ids_matchedfeats_dict, f)






	with open(file_1, 'rb') as f:
		counts_ids_dict = pickle.load(f)


	with open(file_2, 'rb') as f:
		ids_feats_dict = pickle.load(f)

	with open(file_4, 'rb') as f:
		ids_allfeats_dict = pickle.load(f)


	with open(file_5, 'rb') as f:
		ids_matchedfeats_dict = pickle.load(f)

	for key, value in ids_feats_dict.items():
		print(key)
		print(value)

	same_feats = {}

	sub_list = []

	for i in range(1, 6):
		sub_list += counts_ids_dict[i]

	print(len(sub_list))

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

		# print(count)
		count += 1


	final_ids = []

	for i in range(1, 6):
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


		file_name = "new_filtered_ids_" + str(i) + ".txt"

		with open(file_name, 'wb') as f:
			pickle.dump(final_ids, f)

	"""Num matched feats: 5
	Total instances: 21961
	Feature set: 
	frozenset({'vehicle_rates_setpoint', 'battery_status', 'vehicle_attitude_setpoint', 'vehicle_local_position', 'vehicle_gps_position'})
	
	Num matched feats: 6
	Total instances: 18623
	Feature set: 
	frozenset({'vehicle_rates_setpoint', 'battery_status', 'vehicle_attitude_setpoint', 'vehicle_local_position', 'manual_control_setpoint', 'vehicle_gps_position'})
	
	Num matched feats: 7
	Total instances: 14825
	Feature set: 
	frozenset({'vehicle_rates_setpoint', 'manual_control_setpoint', 'battery_status', 'vehicle_attitude_setpoint', 'vehicle_local_position', 'home_position', 'vehicle_gps_position'})
	
	Num matched feats: 8
	Total instances: 9615
	Feature set: 
	frozenset({'vehicle_air_data', 'vehicle_local_position', 'home_position', 'vehicle_gps_position', 'battery_status', 'vehicle_rates_setpoint', 'vehicle_attitude_setpoint', 'manual_control_setpoint'})"""


def grouping_test():
	json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
	with open(json_file, 'r') as inputFile:
		meta_json = json.load(inputFile)
	indexable_meta = dp.get_indexable_meta(meta_json)

	ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
	ulogs_downloaded = os.listdir(ulog_folder)
	ulog_ids = [u[:-4] for u in ulogs_downloaded
				if indexable_meta[u[:-4]]["type"] == "Quadrotor" or indexable_meta[u[:-4]]["type"] == "Fixed Wing"]

	groupings = {}


	filtered_ids = dp.get_filtered_ids(ulog_ids, indexable_meta)
	# filtered_ids = [u for u in filtered_ids if indexable_meta[u]["type"] == "Fixed Wing"]
	# print(len(filtered_ids))

	for u in range(len(filtered_ids)):
		print("Grouping: " + str(u) + "/" + str(len(filtered_ids)))

		ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
		names = dp.convert_to_dfs_ulog(ulog_path, only_col_names=True)      


		combos = set(names) | set(combinations(names, 2)) | set(combinations(names, 3)) | set(combinations(names, 4)) | set(combinations(names, 5)) | set(combinations(names, 6))
		for c in combos:
			if type(c) != str:
				as_set = frozenset(c)
			else:
				as_set = frozenset([c])
			try:
				temp = groupings[as_set]
				groupings[as_set].add(u)
			except:
				groupings[as_set] = set([u])

	# groupings_file = "alltables_ids.txt"

	# with open(groupings_file, "rb") as f:
	#     groupings = pickle.load(f)

	groupings = dict(sorted(groupings.items(), key = lambda x: len(x[1]), reverse=True))

	with open(groupings_file, 'wb') as f:
		pickle.dump(groupings, f)

	count = 0
	for key, value in groupings.items():
		if count < 50:
			print(key)
			counts = Counter([indexable_meta[i]["type"] for i in groupings[key]])
			fixed_percent = round(counts["Fixed Wing"]/(counts["Fixed Wing"] + counts["Quadrotor"]) * 100, 2)
			quad_percent = 100 - fixed_percent
			print("Num Fixed Wing: {} ({}%), Num Quadrotor: {} ({}%)".format(counts["Fixed Wing"], fixed_percent, counts["Quadrotor"], quad_percent))
			print("\n")
			count += 1

def grouping_test_2():
	filtered_ids = dp.get_filtered_ids()

	groupings = {}

	id_feats_dict = {}
	all_feature_names = set()

	with open("ids_allfeats.txt", 'rb') as f:
		ids_allfeats_dict = pickle.load(f)

	# print(len(ids_allfeats_dict))

	for u in range(len(filtered_ids)):
		print("Getting feature names for: " + str(u) + "/" + str(len(filtered_ids)))
		# ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
		try:
			names = ids_allfeats_dict[filtered_ids[u]]  
		except:
			# text file wasn't updated for the new ids so we'll just skip for now
			continue
		
		as_set = frozenset(names)
		all_feature_names = all_feature_names | as_set
		id_feats_dict[filtered_ids[u]] = as_set

	all_feature_names_ls = list(all_feature_names)
	print(len(all_feature_names_ls))


	for f in range(len(all_feature_names_ls)):
		print("Assigning single feat: " + str(f) + "/" + str(len(all_feature_names_ls)))
		feat = all_feature_names_ls[f]

		for u in filtered_ids:
			try:
				if feat in id_feats_dict[u]:
					if feat in groupings:
						groupings[feat].add(u)
						logs_with_27.add(u)
					else:
						groupings[feat] = set([u])
						logs_with_27.add(u)
			except:
				pass



	percentage = 0.6

	# print(len(all_feature_names_ls))
	all_feature_names_ls = [u for u in all_feature_names_ls if len(groupings[u]) >= percentage * len(filtered_ids)]

	print(len(all_feature_names_ls))

	# # logs_with_27 = set()

	# # for key, value in groupings.items():
	# # 	if key in all_feature_names_ls:
	# # 		logs_with_27 = logs_with_27 | groupings[key]

	# # print(list(logs_with_27)[0])
	# # print(len(set(logs_with_27)))

	# # print(len(all_feature_names_ls))

	# curr_ls = []

	# for i in range(len(all_feature_names_ls)):
	# 	print("Assigning pair feat: " + str(i) + "/" + str(len(all_feature_names_ls)))
	# 	for j in range(i + 1, len(all_feature_names_ls)):
	# 		feat_1 = all_feature_names_ls[i]
	# 		feat_2 = all_feature_names_ls[j]
	# 		pairing = set([feat_1, feat_2]) 
	# 		curr_ls.append(pairing)
	# 		groupings[frozenset(pairing)] = groupings[feat_1].intersection(groupings[feat_2])
	


	# subset_sizes = 5


	# for i in range(2, subset_sizes + 1):
	# 	next_ls = []
	# 	visited = {}		
		
	# 	for j in range(len(curr_ls)):
	# 		skipped_ls = list(set(all_feature_names_ls).difference(curr_ls[i]))
	# 		print("Current subset size: " + str(i + 1) + ", Current subset index: " + str(j) + "/" + str(len(curr_ls)))

	# 		for k in range(len(skipped_ls)):
	# 			feat = skipped_ls[k]
	# 			combo = curr_ls[j]
	# 			bigger_combo = set([feat]) | combo

	# 			# print(bigger_combo)

	# 			# if frozenset(bigger_combo) not in visited and frozenset(bigger_combo) not in groupings:
	# 				# visited[frozenset(bigger_combo)] = True
	# 				# next_ls.append(bigger_combo)
	# 				# intersection = groupings[feat].intersection(groupings[frozenset(combo)])

	# 				# if len(intersection) > percentage * len(filtered_ids):
	# 				# 	groupings[frozenset(bigger_combo)] = intersection
	# 			if frozenset(bigger_combo) not in groupings:
	# 				next_ls.append(bigger_combo)
	# 				intersection = groupings[feat].intersection(groupings[frozenset(combo)])
	# 				groupings[frozenset(bigger_combo)] = intersection

	# 	curr_ls = copy.deepcopy(next_ls)

	# 	groupings_file = "alltables_ids_new.txt"
	# 	sorted_groupings = dict(sorted(groupings.items(), key = lambda x: len(x[1]), reverse=True))

	# 	with open(groupings_file, 'wb') as f:
	# 		pickle.dump(sorted_groupings, f)




	groupings_file = "alltables_ids_new.txt"

	with open(groupings_file, "rb") as f:
		groupings = pickle.load(f)


	groupings_with_percents = {}


	count = 0
	for key, value in groupings.items():
		if count < 200000:
			# print(key)
			counts = Counter([indexable_meta[i]["type"] for i in groupings[key]])
			fixed_percent = round(counts["Fixed Wing"]/(counts["Fixed Wing"] + counts["Quadrotor"]) * 100, 2)
			quad_percent = 100 - fixed_percent
			groupings_with_percents[key] = fixed_percent
			# print("Num Fixed Wing: {} ({}%), Num Quadrotor: {} ({}%)".format(counts["Fixed Wing"], fixed_percent, counts["Quadrotor"], quad_percent))
			# print("\n")
			count += 1

	sorted_on_fixed = dict(sorted(groupings_with_percents.items(), key = lambda x: x[1], reverse=True))

	# groupings_file = "alltables_ids_new_sorted.txt"

	# with open(groupings_file, "rb") as f:
	#     groupings = pickle.load(f)

	count = 0
	for key, value in sorted_on_fixed.items():
		if count < 200000:
			print(key)
			counts = Counter([indexable_meta[i]["type"] for i in groupings[key]])
			fixed_percent = round(counts["Fixed Wing"]/(counts["Fixed Wing"] + counts["Quadrotor"]) * 100, 2)
			quad_percent = 100 - fixed_percent
			print("Num Fixed Wing: {} ({}%), Num Quadrotor: {} ({}%)".format(counts["Fixed Wing"], fixed_percent, counts["Quadrotor"], quad_percent))
			print("\n")
			count += 1

def timestamp_shorten_test():
	print("Running ts shorten test")
	with open("ids_matchedfeats.txt", 'rb') as f:
		ids_matchedfeats_dict = pickle.load(f)

	with open("full_parsed_7.txt", 'rb') as f:
		full_parsed = pickle.load(f)

	# full_parsed = dict(islice(full_parsed.items(), 2))
	full_parsed = dp.split_features(full_parsed)

	y = dp.get_labels(list(full_parsed.keys()), indexable_meta)

	percentages = [90, 80, 70, 60, 50]
	for percentage in percentages:
		X = dp.timestamp_bin(full_parsed, keep_percentage=percentage)
		X_data_file = "X_data_7" + "_" + str(percentage) + ".txt"
		with open(X_data_file, 'wb') as f:
			pickle.dump(X, f)   

def timestamp_shorten_bme_test():
	print("Running ts shorten bme test")

	with open("ids_matchedfeats.txt", 'rb') as f:
		ids_matchedfeats_dict = pickle.load(f)

	with open("full_parsed_7.txt", 'rb') as f:
		full_parsed = pickle.load(f)

	# full_parsed = dict(islice(full_parsed.items(), 2))
	full_parsed = dp.split_features(full_parsed)

	y = dp.get_labels(list(full_parsed.keys()), indexable_meta)

	chunks = ["beg", "mid", "end"]
	for chunk in chunks:
		X = dp.timestamp_bin(full_parsed, keep_percentage=33, beg_mid_end=chunk)
		X_data_file = "X_data_7" + "_" + chunk + ".txt"
		with open(X_data_file, 'wb') as f:
			pickle.dump(X, f)   

def timestamp_interval_test():
	print("Running ts interval test")
	ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"

	json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
	with open(json_file, 'r') as inputFile:
		meta_json = json.load(inputFile)
	indexable_meta = dp.get_indexable_meta(meta_json)

	with open("ids_matchedfeats.txt", 'rb') as f:
		ids_matchedfeats_dict = pickle.load(f)

	with open("full_parsed_7.txt", 'rb') as f:
		full_parsed = pickle.load(f)

	# full_parsed = dict(islice(full_parsed.items(), 2))
	full_parsed = dp.split_features(full_parsed)

	y = dp.get_labels(list(full_parsed.keys()), indexable_meta)

	intervals = [750]
	for ints in intervals:
		X = dp.timestamp_bin(full_parsed, num_t_ints=ints)
		X_data_file = "X_data_7" + "_" + str(ints) + "_ints" + ".txt"
		with open(X_data_file, 'wb') as f:
			pickle.dump(X, f) 

def table_drop_test():
	no_home_position = frozenset({'battery_status', 'vehicle_rates_setpoint', 'vehicle_gps_position', 'vehicle_local_position', 'vehicle_attitude_setpoint', 'manual_control_setpoint'})
	no_battery_status = frozenset({'home_position', 'vehicle_rates_setpoint', 'vehicle_gps_position', 'vehicle_local_position', 'vehicle_attitude_setpoint', 'manual_control_setpoint'})
	no_vehicle_rates = frozenset({'home_position', 'battery_status', 'vehicle_gps_position', 'vehicle_local_position', 'vehicle_attitude_setpoint', 'manual_control_setpoint'})
	no_vehicle_local = frozenset({'home_position', 'battery_status', 'vehicle_rates_setpoint', 'vehicle_gps_position', 'vehicle_attitude_setpoint', 'manual_control_setpoint'})
	no_vehicle_attitude = frozenset({'home_position', 'battery_status', 'vehicle_rates_setpoint', 'vehicle_gps_position', 'vehicle_local_position', 'manual_control_setpoint'})
	no_manual_control = frozenset({'home_position', 'battery_status', 'vehicle_rates_setpoint', 'vehicle_gps_position', 'vehicle_local_position', 'vehicle_attitude_setpoint'})
	no_vehicle_gps = frozenset({'home_position', 'battery_status', 'vehicle_rates_setpoint', 'vehicle_local_position', 'vehicle_attitude_setpoint'})

	# original (but also without updated data)
	# Counter({0: 11894, 1: 390})

	# dp.feature_select("no_home", no_home_position)
	# Counter({0: 13898, 1: 471})

	# dp.feature_select("no_batt", no_battery_status)
	# Counter({0: 15897, 1: 513})

	# dp.feature_select("no_rates", no_vehicle_rates)
	# Counter({0: 12887, 1: 412})

	# dp.feature_select("no_local", no_vehicle_local)
	# Counter({0: 13548, 1: 507})

	# dp.feature_select("no_att", no_vehicle_attitude)
	# dp.feature_select("no_man", no_manual_control)  
	# dp.feature_select("no_gps", no_vehicle_gps)

	# dp.preprocess_data("full_parsed_7_no_home.txt", "X_data_7_no_home.txt", "Y_data_7_no_home.txt")
	dp.preprocess_data("full_parsed_7_no_batt.txt", "X_data_7_no_batt.txt", "Y_data_7_no_batt.txt")
	# dp.preprocess_data("full_parsed_7_no_rates.txt", "X_data_7_no_rates.txt", "Y_data_7_no_rates.txt")
	# dp.preprocess_data("full_parsed_7_no_local.txt", "X_data_7_no_local.txt", "Y_data_7_no_local.txt")
	# dp.preprocess_data("full_parsed_7_no_att.txt", "X_data_7_no_att.txt", "Y_data_7_no_att.txt")
	# dp.preprocess_data("full_parsed_7_no_man.txt", "X_data_7_no_man.txt", "Y_data_7_no_man.txt")
	# dp.preprocess_data("full_parsed_7_no_gps.txt", "X_data_7_no_gps.txt", "Y_data_7_no_gps.txt")

def remove_extra_quads(parsed_file, X_orig, y_orig, X_new, y_new):
	json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
	with open(json_file, 'r') as inputFile:
		meta_json = json.load(inputFile)
	indexable_meta = dp.get_indexable_meta(meta_json)

	with open("full_parsed_7.txt", 'rb') as f:
		full_parsed_all_tables = pickle.load(f)
	full_parsed_all_tables = dp.split_features(full_parsed_all_tables)
	kept_quad_ids = [key for key in list(full_parsed_all_tables.keys()) if indexable_meta[key]["type"] == "Quadrotor"]

	with open(parsed_file, 'rb') as f:
		full_parsed_without_table = pickle.load(f)

	full_parsed_without_table = dp.split_features(full_parsed_without_table)

	all_ids = list(full_parsed_without_table.keys())
	kept_indices = []

	for i in range(len(all_ids)):
		if indexable_meta[all_ids[i]]["type"] == "Quadrotor" and all_ids[i] in kept_quad_ids:
			kept_indices.append(i)
		if indexable_meta[all_ids[i]]["type"] == "Fixed Wing":
			kept_indices.append(i)

	
	with open(X_orig, 'rb') as f:
		X_orig = pickle.load(f)

	with open(y_orig, 'rb') as f:
		y_orig = pickle.load(f)

	X_orig = np.array(X_orig)[kept_indices, :, :]
	y_orig = np.array(y_orig)[kept_indices]

	# print(Counter(y_orig))

	with open(X_new, 'wb') as f:
		pickle.dump(X_orig, f)


	with open(y_new, 'wb') as f:
		pickle.dump(y_orig, f)

	# return X, y

def remove_quad_errors(parsed_file, X_orig, y_orig, X_new, y_new):
	json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
	with open(json_file, 'r') as inputFile:
		meta_json = json.load(inputFile)
	indexable_meta = dp.get_indexable_meta(meta_json)


	with open(parsed_file, 'rb') as f:
		full_parsed_all_tables = pickle.load(f)
	full_parsed_all_tables = dp.split_features(full_parsed_all_tables)
	kept_quad_ids = [key for key in list(full_parsed_all_tables.keys()) if indexable_meta[key]["type"] == "Quadrotor" and indexable_meta[key]["errors"] == 0]

	all_ids = list(full_parsed_all_tables.keys())
	kept_indices = []

	for i in range(len(all_ids)):
		if indexable_meta[all_ids[i]]["type"] == "Quadrotor" and all_ids[i] in kept_quad_ids:
			kept_indices.append(i)
		if indexable_meta[all_ids[i]]["type"] == "Fixed Wing":
			kept_indices.append(i)

	
	with open(X_orig, 'rb') as f:
		X_orig = pickle.load(f)

	with open(y_orig, 'rb') as f:
		y_orig = pickle.load(f)

	print(Counter(y_orig))

	X_orig = np.array(X_orig)[kept_indices, :, :]
	y_orig = np.array(y_orig)[kept_indices]

	print(Counter(y_orig))

	with open(X_new, 'wb') as f:
		pickle.dump(X_orig, f)


	with open(y_new, 'wb') as f:
		pickle.dump(y_orig, f)

	# return X, y



def run_multiprocessing(left_bound):
	print("running multiprocessing")
	num_processes = 30
	num_feats_per_proc = 5

	ranges = []
	for i in range(num_processes):
		ranges.append([left_bound, left_bound + num_feats_per_proc])
		left_bound = left_bound + num_feats_per_proc

	pool = Pool(processes=len(ranges))
	pool.map(pickling_dataframes_test, ranges)



def pickling_dataframes_test(bounds):
	filtered_ids = dp.get_filtered_ids()
	# print(len(filtered_ids))
	feats = {}
	sanity_check = 0
	with open(work_dir + "all_feats_in_60.txt", 'rb') as f:
		all_feature_names_ls = pickle.load(f)

	with open(work_dir + "ids_allfeats.txt", "rb") as f:
		ids_allfeats = pickle.load(f)

	# print(len(all_feature_names_ls))

	# print(len(all_feature_names_ls))

	# print(len(filtered_ids[left_bound:right_bound]))
	# print(filtered_ids[left_bound])
	# print(filtered_ids[right_bound - 1])

	# for u in range(left_bound, right_bound):
	# 	id = filtered_ids[u]
	# 	ulog_path = os.path.join(ulog_folder, id + ".ulg")
	# 	dfs, names = dp.convert_to_dfs_ulog(ulog_path)
	# 	all_feats = ids_allfeats[id]

	# 	print(u)

	# 	for i in range(len(all_feature_names_ls)):
	# 		feat_name = all_feature_names_ls[i]

	# 		if feat_name in all_feats:				
	# 			col = extract_individual(dfs, feats, feats_subset=[feat_name])[feat_name]	
	# 		else:
	# 			col = {}


	# 		if feat_name in feats.keys():
	# 			feats[feat_name][id] = col
	# 		else:
	# 			feats[feat_name] = {}
	# 			feats[feat_name][id] = col

	# with open(work_dir + "feats_" + str(ind) + ".txt", 'wb') as f:
	# 	pickle.dump(feats, f)

	# print(dp.convert_to_dfs_ulog(os.path.join(ulog_folder, "00745da2-554c-4358-936f-f090159aad08.ulg")))
	# print(dp.convert_to_dfs_csv(os.path.join(csv_folder, "00745da2-554c-4358-936f-f090159aad08")))


	for i in range(bounds[0], bounds[1]):
		feat_name = all_feature_names_ls[i]
		feature_file_name = work_dir + "feature_maps/" + feat_name + ".txt"

		if os.path.exists(feature_file_name):
			print(feature_file_name + " already exists, skipping")
			continue

		feats = {}
		

		for u in range(len(filtered_ids)):
			# raw_ulog = False

			# id = filtered_ids[u]
			# csv_path = os.path.join(csv_folder, id)
			# dfs, names = dp.convert_to_dfs_csv(csv_path)

			# if len(dfs.keys()) == 0:
			# 	raw_ulog = True
			# 	ulog_path = os.path.join(ulog_folder, id + ".ulg")
			# 	dfs, names = dp.convert_to_dfs_ulog(ulog_path)

			id = filtered_ids[u]
			ulog_path = os.path.join(ulog_folder, id + ".ulg")
			dfs, names = dp.convert_to_dfs_ulog(ulog_path)

			all_feats = ids_allfeats[id]

			if feat_name in all_feats:	
				col = extract_individual(dfs, feats, feats_subset=[feat_name])[feat_name]	
			else:
				col = {}


			if feat_name in feats.keys():
				feats[feat_name][id] = col
			else:
				feats[feat_name] = {}
				feats[feat_name][id] = col

			print("Feat: " + str(i) + ", Log: " + str(u) + "/" + str(len(filtered_ids)))			

		with open(feature_file_name, 'wb') as f:
			pickle.dump(feats[feat_name], f)



	# for i in range(len(all_feature_names_ls)): 
	# 	feat_name = all_feature_names_ls[i]
	# 	with open(work_dir + feat_name + ".txt", 'wb') as f:
	# 		pickle.dump(feats[feat_name], f)

	# for i in range(len(all_feature_names_ls)): 
	# 	feat_name = all_feature_names_ls[i]
	# 	print(len(feats[feat_name]))
	# 	os.remove(feat_name)



		# if u == round(len(filtered_ids)/5) - 1:
		# 	file_1 = work_dir + "ids_dfs_1.txt"

		# 	with open(file_1, 'wb') as f:
		# 		pickle.dump(ids_dfs, f)
		# 	sanity_check += len(ids_dfs)
		# 	ids_dfs = {}

# def aggregate_pickles():
# 	with open(work_dir + "all_feats.txt", 'rb') as f:
# 		all_feature_names_ls = pickle.load(f)



# 	with open(work_dir + "feats_1.txt", 'rb') as f:
# 		d_1 = pickle.load(f)

# 	with open(work_dir + "feats_2.txt", 'rb') as f:
# 		d_2 = pickle.load(f)	

# 	with open(work_dir + "feats_3.txt", 'rb') as f:
# 		d_3 = pickle.load(f)	

# 	with open(work_dir + "feats_4.txt", 'rb') as f:
# 		d_4 = pickle.load(f)	

# 	with open(work_dir + "feats_5.txt", 'rb') as f:
# 		d_5 = pickle.load(f)	

# 	data = [d_1, d_2, d_3, d_4, d_5]

# 	for i in range(len(all_feature_names_ls)): 
# 		feat_df = {}
# 		feat_name = all_feature_names_ls[i]
# 		feat_df[feat_name] = {}
# 		for d in data:
# 			feat_df[feat_name].update(d[feat_name])

def saving_alltables():
	filtered_ids = dp.get_filtered_ids()
	ids_allfeats = {}

	for u in range(len(filtered_ids)):
		id = filtered_ids[u]
		ulog_path = os.path.join(ulog_folder, id + ".ulg")

		names = dp.convert_to_dfs_ulog(ulog_path, only_col_names=True)

		ids_allfeats[id] = names

		print("Log: " + str(u) + "/" + str(len(filtered_ids)))

	with open(work_dir + "ids_allfeats.txt", 'wb') as f:
		pickle.dump(ids_allfeats, f)

	ids_allfeats = {}	

def checking_multiple_flights():
	# id = "60162ea8-0ee6-4ed7-9e3e-bfc91242420a"
	# ulog_path = os.path.join(ulog_folder, id + ".ulg")

	# dfs, names = dp.convert_to_dfs_ulog(ulog_path)
	# print(dfs)
	filtered_ids = dp.get_filtered_ids()
	new_filtered_ids = []

	for u in range(len(filtered_ids)):
		id = filtered_ids[u]
		ulog_path = os.path.join(ulog_folder, id + ".ulg")

		dfs, names = dp.convert_to_dfs_ulog(ulog_path, specific_tables=["vehicle_status"])

		if len(dfs) != 0:
			new_filtered_ids.append(dfs)


	print(len(new_filtered_ids))


def extract_individual(dfs, feats, feats_subset):
	feature_dict = {}
	for feat in feats_subset:
		strings = feat.split(" ")
		table_name = strings[0]
		feat_name = strings[2]
		# print(feat_name)

		try:
			feature_dict[feat] = [dfs[table_name][0][["timestamp"] + [feat_name]]]
		except:
			# strings = feat_name.split(" ")
			# if strings[0] in ["previous", "next", "current"]:
			# 	csv_feat_name = feat_name.replace("_", ".", 1)
			# else:
			# 	print(table_name)
			# 	print(feat_name)
			# csv_feat_name = feat_name

			print(table_name)
			print(feat_name)

			extracted_df = dfs[table_name][0]
			# print(extracted_df.columns)
			extracted_df.columns = extracted_df.columns.str.replace("[", "_")
			extracted_df.columns = extracted_df.columns.str.replace("]", "_")
			extracted_df.columns = extracted_df.columns.str.replace(".", "_")
			# print(extracted_df.columns)

			extracted_df = extracted_df[["timestamp"] + [feat_name]]
			# extracted_df.rename({csv_feat_name: feat_name}, axis=1, inplace=True)

			feature_dict[feat] = [extracted_df]

	return feature_dict

# def parse_files(feats_subset):
# 	full_parsed = {}
# 	filtered_ids = dp.get_filtered_ids()

# 	# print(len(set(ids_allfeats.keys()).intersection(set(filtered_ids))))

# 	gc.disable()
# 	with open(work_dir + "ids_dfs_1_fast.txt", 'rb') as f:
# 		ids_dfs = cpickle.load(f)
# 	f.close()
# 	gc.enable()

# 	# print(len(set(ids_dfs.keys()).intersection(set(filtered_ids))))

# 	# print(len(set(ids_dfs.keys()).intersection(set(ids_allfeats.keys()))))

# 	count = 0


# 	for u in range(len(filtered_ids)):
# 		if u == 2*round(len(filtered_ids)/5) - 1:
# 			with open(work_dir + "ids_dfs_2.txt", 'rb') as f:
# 				ids_dfs = pickle.load(f)
# 			count = 0

# 		if u == 3*round(len(filtered_ids)/5) - 1:
# 			with open(work_dir + "ids_dfs_3.txt", 'rb') as f:
# 				ids_dfs = pickle.load(f)
# 			count = 0


# 		if u == 4*round(len(filtered_ids)/5) - 1:
# 			with open(work_dir + "ids_dfs_4.txt", 'rb') as f:
# 				ids_dfs = pickle.load(f)
# 			count = 0

# 		if u == len(filtered_ids) - 1:
# 			with open(work_dir + "ids_dfs_5.txt", 'rb') as f:
# 				ids_dfs = pickle.load(f)
# 			count = 0

# 		log_id = filtered_ids[count]
# 		if u % 200 == 0:
# 			print("Log: " + str(u) + "/" + str(len(filtered_ids)))
# 		if  log_id not in ids_allfeats.keys() or log_id not in ids_dfs.keys():
# 			continue

# 		dfs = ids_dfs[log_id]
# 		feats = ids_allfeats[log_id]

# 		# dfs, names = dp.convert_to_dfs_ulog(ulog_path)
# 		# feats = dfs['vehicle_local_position'][0].keys()
# 		# feats = ['vehicle_local_position | ' + f for f in feats]
# 		# print(feats)
# 		# print(feats_subset)


# 		if len(set(feats).intersection(feats_subset)) == len(feats_subset):
# 			feature_dict = extract_individual(dfs, feats, feats_subset=feats_subset)
# 			# print(feature_dict)
# 			full_parsed[filtered_ids[u]] = feature_dict

# 		count += 1

# 	return full_parsed

def parse_files(feats_subset):
	filtered_ids = dp.get_filtered_ids()
	full_parsed = {}

	for feat in feats_subset:

		file_dir = work_dir + "feature_maps/" + feat + ".txt"

		with open(file_dir, "rb") as f:
			mapping = pickle.load(f)

		for u in range(len(filtered_ids)): 

			if filtered_ids[u] in mapping and len(mapping[filtered_ids[u]]) != 0:

				if filtered_ids[u] not in full_parsed.keys():
					full_parsed[filtered_ids[u]] = {feat: mapping[filtered_ids[u]]}		
				else:
					full_parsed[filtered_ids[u]][feat] = mapping[filtered_ids[u]]

	# print(full_parsed['fe8f339d-cad0-4d42-ac59-fe14b26d419d'])

	print(len(full_parsed))
	return full_parsed


def store_X_and_Y(subset_dir, parsed_file, X_file, y_file):
	with open(subset_dir + "/" + parsed_file, 'rb') as f:
		parsing = pickle.load(f)
	
	full_parsed_split = dp.split_features(parsing)

	y = dp.get_labels(list(parsing.keys()), indexable_meta)

	X = dp.timestamp_bin(full_parsed_split)

	with open(X_file, 'wb') as f:
		pickle.dump(X, f)

	with open(y_file, 'wb') as f:
		pickle.dump(y, f)

def get_tables_in_60():
	filtered_ids = dp.get_filtered_ids()
	groupings = {}
	id_feats_dict = {}
	all_table_names = set()

	# print(len(ids_allfeats_dict))

	for u in range(len(filtered_ids)):
		# print("Getting feature names for: " + str(u) + "/" + str(len(filtered_ids)))
		# ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
		try:
			names = ids_allfeats_dict[filtered_ids[u]]  
			if type(names) is not list:
				continue
		except:
			# text file wasn't updated for the new ids so we'll just skip for now
			continue


		as_set = frozenset(names)
		all_table_names = all_table_names | as_set
		id_feats_dict[filtered_ids[u]] = as_set

	all_table_names_ls = list(all_table_names)


	for f in range(len(all_table_names_ls)):
		# print("Assigning single feat: " + str(f) + "/" + str(len(all_feature_names_ls)))
		feat = all_table_names_ls[f]

		for u in filtered_ids:
			try:
				if feat in id_feats_dict[u]:
					if feat in groupings:
						groupings[feat].add(u)
						logs_with_27.add(u)
					else:
						groupings[feat] = set([u])
						logs_with_27.add(u)
			except:
				pass

	percentage = 0.6

	all_table_names_ls = [u for u in all_table_names_ls if len(groupings[u]) >= percentage * len(filtered_ids)]

	return all_table_names_ls


def get_features_in_60():
	with open(work_dir + "all_feats_in_topic_subset.txt", 'rb') as f:
		all_feature_names_ls = pickle.load(f)

	with open(work_dir + "ids_allfeats_in_topic_subset.txt", 'rb') as f:
		ids_allfeats = pickle.load(f)	

	threshold = len(dp.get_filtered_ids()) * 0.60
	over_60_count = 0
	feat_counts = {}
	feats_list = []



	for feat in all_feature_names_ls:
		for key, value in ids_allfeats.items():
			if feat in value:
				if feat in feat_counts.keys():
					feat_counts[feat] += 1
				else:
					feat_counts[feat] = 1
		if feat_counts[feat] >= threshold:
			feats_list.append(feat)
			print(len(feats_list))

	print(len(feats_list))

	with open(work_dir + "all_feats_in_60_in_topic_subset.txt", 'wb') as f:
		pickle.dump(feats_list, f)

	return feats_list
	



def saving_allfeats():
	filtered_ids = dp.get_filtered_ids()
	all_feature_names_ls = get_features_in_60()
	all_table_names_ls = ["vehicle_local_position", "sensor_gps", "vehicle_gps_position", "vehicle_attitude_setpoint", "vehicle_rates_setpoint", "rollspeed",
					"home_position", "vehicle_attitude_setpoint", "sensor_accel", "sensor_gyro", "sensor_mag", "vehicle_air_data", "manual_control_setpoint", "battery_status"]
	

	ids_allfeats = {}

	all_feature_names = set()
	

	for i in range(len(filtered_ids)):
		log_feature_names = set()
		id = filtered_ids[i]
		if indexable_meta[id]["type"] == "Hexarotor":
			ulog_path = os.path.join(ulog_folder_hex, id + ".ulg")
		else:
			ulog_path = os.path.join(ulog_folder, id + ".ulg")		

		dfs, names = dp.convert_to_dfs_ulog(ulog_path)

		for j in range(len(all_table_names_ls)):
			try:
				feats = [all_table_names_ls[j] + " | " + k 
						 for k in list(dfs[all_table_names_ls[j]][0].keys())
						 if k != "timestamp" and k != 'ref_timestamp' and k != "surface_bottom_timestamp"]

				feats = [f for f in feats if f in all_feature_names_ls]
				as_set = frozenset(feats)
				log_feature_names = log_feature_names | as_set
				all_feature_names = all_feature_names | as_set

			except:
				# print("table does not exist")
				continue

		print(len(all_feature_names))

		ids_allfeats[filtered_ids[i]] = list(log_feature_names)

		print("Log: " + str(i) + "/" + str(len(filtered_ids)))

	with open(work_dir + "all_feats_in_topic_subset.txt", 'wb') as f:
		pickle.dump(list(all_feature_names), f)

	with open(work_dir + "ids_allfeats_in_topic_subset.txt", 'wb') as f:
		pickle.dump(ids_allfeats, f)	

def get_feat_subsets():
	baseline_feats = ["vehicle_local_position | x", "vehicle_local_position | y",
					 "vehicle_local_position | z", "vehicle_attitude_setpoint | roll_body",
					 "vehicle_attitude_setpoint | pitch_body", "vehicle_attitude_setpoint | yaw_body"]

	with open(work_dir + "all_feats_in_topic_subset.txt", 'rb') as f:
		all_feats = pickle.load(f)	

	print(len(all_feats))

	parsed_feats = [f for f in all_feats if f.split(" ")[0] != "battery_status"]

	parsed_feats = [f for f in parsed_feats if not any(char.isdigit() for char in f)]

	print(len(parsed_feats))

	# grouped_feats = {}
	# for feat in all_feats:
	# 	if feat in existing_feats:
	# 		print("true")
	# 	topic = feat.split(" ")[0]
	# 	feat_name = feat.split(" ")[2]
	# 	if topic not in grouped_feats.keys():
	# 		grouped_feats[topic] = [feat_name]
	# 	else:
	# 		grouped_feats[topic].append(feat_name)

	# print(json.dumps(grouped_feats, indent=2))



	all_directional_feats = []
	new_parsed_feats = []

	for full in parsed_feats:
		if full not in baseline_feats:
			full_split = full.split(" ")
			feat_split = full_split[2].split("_")

			if "yaw" in feat_split or "roll" in feat_split or "pitch" in feat_split or "x" in feat_split or "y" in feat_split or "z" in feat_split:
				all_directional_feats.append(full)
			else:
				new_parsed_feats.append(full)



	potential_directional_feats = {"home_position | x": ["home_position | y", "home_position | z"],
						 "manual_control_setpoint | x": ["manual_control_setpoint | y", "manual_control_setpoint | z"],
						 "vehicle_rates_setpoint | yaw": ["vehicle_rates_setpoint | roll", "vehicle_rates_setpoint | pitch"],
						 "vehicle_attitude_setpoint | yaw_reset_integral": ["vehicle_attitude_setpoint | roll_reset_integral", "vehicle_attitude_setpoint | pitch_reset_integral"]}


	for feat in list(potential_directional_feats.keys()):
		new_parsed_feats.append(feat)


	"""
	other potential directional_feats
	vehicle_attitude_setpoint | fw_control_yaw
	vehicle_local_position | z_reset_counter
	vehicle_local_position | z_valid
	home_position | yaw
	vehicle_local_position | v_z_valid
	vehicle_local_position | z_global
	vehicle_attitude_setpoint | yaw_sp_move_rate
	vehicle_local_position | delta_z
	vehicle_local_position | z_deriv
	"""

	all_feat_subsets = {}

	for j in range(5):
		feat_subset = []
		for i in range(5):
			randint = random.randint(0, len(new_parsed_feats) - 1)
			while new_parsed_feats[randint] in feat_subset and new_parsed_feats[randint] in baseline_feats:
				randint = random.randint(0, len(new_parsed_feats) - 1)

			feat_subset.append(new_parsed_feats[randint])

			if new_parsed_feats[randint] in potential_directional_feats:
				feat_subset += potential_directional_feats[new_parsed_feats[randint]]

		feat_subset += baseline_feats
		all_feat_subsets[j] = feat_subset
		print(feat_subset)
		print("\n")


	with open(work_dir + "feat_subsets.txt", 'wb') as f:
		pickle.dump(all_feat_subsets, f)	


def drop_fake_features():
	with open(work_dir + "all_feats.txt", 'rb') as f:
		all_feature_names_ls = pickle.load(f)

	print(len(all_feature_names_ls))
	for feat in all_feature_names_ls: 
		if feat.split(" ")[2] == "next_timestamp":
			all_feature_names_ls.remove(feat)

	print(len(all_feature_names_ls))

	with open(work_dir + "all_feats.txt", 'wb') as f:
		pickle.dump(all_feature_names_ls, f)

def brute_force_feat_select(num_features, best_subset=[]):
	with open(work_dir + "all_feats_in_60.txt", 'rb') as f:
		all_feature_names_ls = pickle.load(f)

	print(len(all_feature_names_ls))

	# all_feature_names_ls = [all_feature_names_ls[0]]


	subset_dir = work_dir + num_features + "_feature_parsings"

	if not os.path.isdir(subset_dir):
		os.makedirs(subset_dir)

	feat_map_dir = work_dir + "feature_maps"
	
	available_features = [x[:-4] for x in os.listdir(feat_map_dir)]


	# for i in range(len(available_features)):
	# 	if available_features[i] in all_feature_names_ls:
	# 		print("Feat: " + str(i) + "/" + str(len(available_features)))
	# 		parsed_file = ""
	# 		if best_subset == []:
	# 			feats_subset = [available_features[i]]
	# 			parsed_file = "full_parsed_feat_" + available_features[i] + ".txt"
	# 			full_parsed = parse_files(feats_subset)
	# 	else:
	# 		concat_string = ""
	# 		feats_subset = []
	# 		for j in range(len(best_subset)):
	# 			feats_subset.append(all_feature_names_ls[best_subset[j]])
	# 			concat_string += str(best_subset[j])
	# 			concat_string += "_"

	# 		feats_subset.append(all_feature_names_ls[i])
	# 		parsed_file = "full_parsed_feat_" + concat_string + str(i) + ".txt"
	# 		full_parsed = parse_files(feats_subset)


			# with open(subset_dir + "/" + parsed_file, 'wb') as f:
			# 	pickle.dump(full_parsed, f)


	# print(len(os.listdir(work_dir + "one_feature_parsings")))


	for i in range(len(available_features)):

		if available_features[i] in all_feature_names_ls:
			concat_string = available_features[i]
			# for j in range(len(best_subset)):
			# 	concat_string += str(best_subset[j])
			# 	concat_string += "_"
			parsed_file = "full_parsed_feat_" + concat_string + ".txt"
			X_file = subset_dir + "/" + "X_data_feat_" + concat_string + ".txt"
			y_file = subset_dir + "/" + "y_data_feat_" + concat_string + ".txt"
			store_X_and_Y(subset_dir, parsed_file, X_file, y_file)	


def feature_selection_justification():
	filtered_ids = dp.get_filtered_ids()

	groupings = {}

	id_feats_dict = {}
	all_feature_names = set()

	with open("ids_allfeats.txt", 'rb') as f:
		ids_allfeats_dict = pickle.load(f)

	# print(len(ids_allfeats_dict))

	for u in range(len(filtered_ids)):
		# print("Getting feature names for: " + str(u) + "/" + str(len(filtered_ids)))
		# ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
		try:
			names = ids_allfeats_dict[filtered_ids[u]]  
		except:
			# text file wasn't updated for the new ids so we'll just skip for now
			continue
		
		as_set = frozenset(names)
		all_feature_names = all_feature_names | as_set
		id_feats_dict[filtered_ids[u]] = as_set

	all_feature_names_ls = list(all_feature_names)

	for f in range(len(all_feature_names_ls)):
		# print("Assigning single feat: " + str(f) + "/" + str(len(all_feature_names_ls)))
		feat = all_feature_names_ls[f]

		for u in filtered_ids:
			try:
				if feat in id_feats_dict[u]:
					if feat in groupings:
						groupings[feat].add(u)
					else:
						groupings[feat] = set([u])
			except:
				pass

	print(len(all_feature_names_ls))
	print(len(filtered_ids))


	percentage = 0.1

	all_feature_names_ls = [u for u in all_feature_names_ls if len(groupings[u]) >= percentage * len(filtered_ids)]	
	

	logs_with_27 = set()
	for f in range(len(all_feature_names_ls)):
		# print("Assigning single feat: " + str(f) + "/" + str(len(all_feature_names_ls)))
		feat = all_feature_names_ls[f]

		for u in filtered_ids:
			try:
				if feat in id_feats_dict[u]:
					if feat in groupings:
						groupings[feat].add(u)
						logs_with_27.add(u)
					else:
						groupings[feat] = set([u])
						logs_with_27.add(u)
			except:
				pass
	print(len(all_feature_names_ls))
	print(len(logs_with_27))



def save_durations():
	# with open(work_dir + "durations.txt", 'rb') as f:
	# 	old_durations = pickle.load(f)
	# id = "00d29a7c-aae8-4f50-ae8b-9befe5c69e1e"
	# ulog_path = os.path.join(ulog_folder, id + ".ulg")
	# log = dp.pyulog.ULog(ulog_path)
	# print(log.start_timestamp)
	# print(log.last_timestamp)
	# print([msg.name for msg in log.data_list if msg.name == 'vehicle_local_position_setpoint'])
	# log.data_list = [msg.name for msg in log.data_list if msg.name == 'vehicle_local_position_setpoint']
	# print(log.start_timestamp)

	# print((log.last_timestamp - log.start_timestamp)/1e6)

	# for key, value in durations.items():
	# 	if value/1e6 > 1500:
	# 		ulog_path = os.path.join(ulog_folder, key + ".ulg")
	# 		log = dp.pyulog.ULog(ulog_path)
	# 		print((value/1e6, (log.last_timestamp - log.start_timestamp)/1e6))

	filtered_ids = dp.get_filtered_ids()
	durations = {}

	for u in range(len(filtered_ids)):
		id = filtered_ids[u]
		# print(old_durations[id])
		if indexable_meta[id]["type"] == "Hexarotor":
			ulog_path = os.path.join(ulog_folder_hex, id + ".ulg")
		else:
			ulog_path = os.path.join(ulog_folder, id + ".ulg")
		try:
			log = dp.pyulog.ULog(ulog_path)
			durations[id] = [log.start_timestamp, log.last_timestamp]

		except:

			durations[id] = "parsing error"

		print("Log: " + str(u) + "/" + str(len(filtered_ids)))	


		
	with open(work_dir + "new_durations.txt", 'wb') as f:
		pickle.dump(durations, f)


def check_status():
	with open(work_dir + "all_feats_in_60.txt", 'rb') as f:
		all_feature_names_ls = pickle.load(f)

	current_feats = os.listdir(work_dir + "feature_maps/")
	current_feats = list(map(lambda x : x[:-4], current_feats))

	print(len(set(current_feats).intersection(set(all_feature_names_ls))))

def get_tarik_data():
	file = "../../../work/uav-ml/official_experiments/sampling/X_sampling_1_50_ints.txt"
	remapped = {}
	with open(file, "rb") as f:
		X_data = pickle.load(f)

	for i, value in enumerate(X_data):
		remapped[temp_ids[i]] = value

	with open("mapped_processed_data.txt", "wb") as f:
		pickle.dump(remapped, f)



def main():
	# grouping_test()

	# grouping_test_2()

	# timestamp_shorten_test()

	# timestamp_shorten_bme_test()

	# timestamp_interval_test()

	# table_drop_test()

	# checking_multiple_flights()

	# feature_selection_justification()

	# pickling_dataframes_test([1, 10])

	# saving_alltables()	
	# saving_allfeats()
	# brute_force_feat_select("one")
	# brute_force_feat_select("two", [0, 1])

	# check_pickle()

	# remove_extra_quads("full_parsed_7_no_batt.txt", "X_data_7_no_batt.txt", "Y_data_7_no_batt.txt", "X_data_7_no_batt_no_extra.txt", "Y_data_7_no_batt_no_extra.txt")

	# parse_files(["vehicle_local_position | x"])

	# remove_quad_errors("full_parsed_7.txt", "X_data_7.txt", "Y_data_7.txt", "X_data_7_no_quad_errors.txt", "Y_data_7_no_quad_errors.txt")

	# drop_fake_features()

	# run_multiprocessing(0)
	# run_multiprocessing(150)

	# parse_files(["battery_status | full_charge_capacity_wh"])

	# save_durations()

	# get_features_in_60()

	# check_status()

	# dp.get_filtered_ids()

	# dp.preprocess_data("X_data_7_multi.txt", "Y_data_7_multi.txt")

	# get_feat_subsets()

	# dp.preprocess_data("X_data_7_set_1.txt", "Y_data_7_set_1.txt")

	# with open(work_dir + "feat_subsets.txt", "rb") as f:
	# 	feat_subsets = pickle.load(f)

	# print(feat_subsets)





	# with open(work_dir + "official_experiments/feat_select/Y_data_feat_subset_2.txt", "rb") as f:
	# with open("Y_data_7.txt", "rb") as f:
	# 	y = pickle.load(f)


	# dp.preprocess_data("X_data_feat_subset_baseline.txt", "Y_data_feat_subset_baseline.txt", parse_id="feat_subset_baseline", feats_subset=baseline_feats)


	# print(Counter(y))

	# feat_subset = feat_subsets[3]
	# dp.preprocess_data("X_data_feat_subset_3.txt", "Y_data_feat_subset_3.txt", parse_id="feat_subset_3", feats_subset=feat_subset)

	# with open(work_dir + "official_experiments/feat_select/" + "full_parsed_feat_subset_0.txt", "rb") as f:
	# 	X = pickle.load(f)

	# print(X[list(X.keys())[0]])

	# print(X[0][0])
	# print(np.array(X).shape)

	get_tarik_data()


if __name__ == "__main__":
	main()
