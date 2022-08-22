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

	id_feats_dict = {}
	all_feature_names = set()

	with open("ids_allfeats.txt", 'rb') as f:
		ids_allfeats_dict = pickle.load(f)

	for u in range(len(filtered_ids)):
		print("Getting feature names for: " + str(u) + "/" + str(len(filtered_ids)))
		# ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
		names = ids_allfeats_dict[filtered_ids[u]]  
		
		as_set = frozenset(names)
		all_feature_names = all_feature_names | as_set
		id_feats_dict[filtered_ids[u]] = as_set

	all_feature_names_ls = list(all_feature_names)

	for f in range(len(all_feature_names_ls)):
		print("Assigning single feat: " + str(f) + "/" + str(len(all_feature_names_ls)))
		feat = all_feature_names_ls[f]

		for u in filtered_ids:
			if feat in id_feats_dict[u]:
				if feat in groupings:
					groupings[feat].add(u)
				else:
					groupings[feat] = set([u])
	

	curr_ls = []

	for i in range(len(all_feature_names_ls)):
		print("Assigning pair feat: " + str(i) + "/" + str(len(all_feature_names_ls)))
		for j in range(i + 1, len(all_feature_names_ls)):
			feat_1 = all_feature_names_ls[i]
			feat_2 = all_feature_names_ls[j]
			pairing = set([feat_1, feat_2]) 
			curr_ls.append(pairing)
			groupings[frozenset(pairing)] = groupings[feat_1].intersection(groupings[feat_2])
	


	subset_sizes = 5


	for i in range(2, subset_sizes + 1):
		next_ls = []
		
		
		for i in range(len(curr_ls)):
			skipped_ls = list(set(all_feature_names_ls).difference(curr_ls[i]))
			print("Current subset size: " + str(i)) + ", Current subset index: " + str(i) + "/" + str(len(curr_ls))

			for j in range(len(skipped_ls)):
				feat = skipped_ls[j]
				combo = curr_ls[i]
				
				bigger_combo = set([feat]) | combo
				# print(bigger_combo)

				if frozenset(bigger_combo) not in groupings:
					next_ls.append(bigger_combo)
					groupings[frozenset(bigger_combo)] = groupings[feat].intersection(groupings[frozenset(combo)])

		curr_ls = copy.deepcopy(next_ls)


	groupings_file = "alltables_ids.txt"

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

def timestamp_shorten_test():
	print("Running ts shorten test")
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

	percentages = [90, 80, 70, 60, 50]
	for percentage in percentages:
		X = dp.timestamp_bin(full_parsed, keep_percentage=percentage)
		X_data_file = "X_data_7" + "_" + str(percentage) + ".txt"
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

	intervals = [100, 200, 500]
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

	dp.feature_select("no_home", no_home_position)
	dp.feature_select("no_batt", no_battery_status)
	dp.feature_select("no_rates", no_vehicle_rates)
	dp.feature_select("no_local", no_vehicle_local)
	dp.feature_select("no_att", no_vehicle_attitude)
	dp.feature_select("no_man", no_manual_control)  

def main():
	# grouping_test()

	grouping_test_2()

	# timestamp_shorten_test()

	# timestamp_interval_test()

	# table_drop_test()

	

if __name__ == "__main__":
	main()
