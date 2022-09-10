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

ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
work_dir = "../../../work/uav-ml/"

json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
with open(json_file, 'r') as inputFile:
	meta_json = json.load(inputFile)
indexable_meta = dp.get_indexable_meta(meta_json)

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
	# filtered_ids = dp.get_filtered_ids()

	# groupings = {}

	# id_feats_dict = {}
	# all_feature_names = set()

	# with open("ids_allfeats.txt", 'rb') as f:
	# 	ids_allfeats_dict = pickle.load(f)

	# # print(len(ids_allfeats_dict))

	# for u in range(len(filtered_ids)):
	# 	print("Getting feature names for: " + str(u) + "/" + str(len(filtered_ids)))
	# 	# ulog_path = os.path.join(ulog_folder, filtered_ids[u] + ".ulg")
	# 	try:
	# 		names = ids_allfeats_dict[filtered_ids[u]]  
	# 	except:
	# 		# text file wasn't updated for the new ids so we'll just skip for now
	# 		continue
		
	# 	as_set = frozenset(names)
	# 	all_feature_names = all_feature_names | as_set
	# 	id_feats_dict[filtered_ids[u]] = as_set

	# all_feature_names_ls = list(all_feature_names)


	# for f in range(len(all_feature_names_ls)):
	# 	print("Assigning single feat: " + str(f) + "/" + str(len(all_feature_names_ls)))
	# 	feat = all_feature_names_ls[f]

	# 	for u in filtered_ids:
	# 		try:
	# 			if feat in id_feats_dict[u]:
	# 				if feat in groupings:
	# 					groupings[feat].add(u)
	# 					logs_with_27.add(u)
	# 				else:
	# 					groupings[feat] = set([u])
	# 					logs_with_27.add(u)
	# 		except:
	# 			pass



	# percentage = 0.6

	# # print(len(all_feature_names_ls))
	# all_feature_names_ls = [u for u in all_feature_names_ls if len(groupings[u]) >= percentage * len(filtered_ids)]

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

	intervals = [500, 2000]
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
	dp.preprocess_data("full_parsed_7_no_att.txt", "X_data_7_no_att.txt", "Y_data_7_no_att.txt")
	# dp.preprocess_data("full_parsed_7_no_man.txt", "X_data_7_no_man.txt", "Y_data_7_no_man.txt")
	# dp.preprocess_data("full_parsed_7_no_gps.txt", "X_data_7_no_gps.txt", "Y_data_7_no_gps.txt")


def pickling_dataframes_test():
	filtered_ids = dp.get_filtered_ids()
	ids_dfs = {}
	print(len(filtered_ids))
	sanity_check = 0

	for u in range(len(filtered_ids)):
		id = filtered_ids[u]
		ulog_path = os.path.join(ulog_folder, id + ".ulg")
		dfs, names = dp.convert_to_dfs_ulog(ulog_path)
		ids_dfs[id] = dfs	


		if u == round(len(filtered_ids)/3) - 1:
			file_1 = work_dir + "ids_dfs_1.txt"

			with open(file_1, 'wb') as f:
				pickle.dump(ids_dfs, f)
			sanity_check += len(ids_dfs)
			ids_dfs = {}

		# if u > round(len(filtered_ids)/3) - 1:
		
		if u == 2*round(len(filtered_ids)/3) - 1:
			file_2 = work_dir + "ids_dfs_2.txt"

			with open(file_2, 'wb') as f:
				pickle.dump(ids_dfs, f)
			sanity_check += len(ids_dfs)
			ids_dfs = {}

		if u == len(filtered_ids) - 1:
			file_3 = work_dir + "ids_dfs_3.txt"

			with open(file_3, 'wb') as f:
				pickle.dump(ids_dfs, f)
			sanity_check += len(ids_dfs)
			ids_dfs = {}

		print("Pickled: " + str(u) + "/" + str(len(filtered_ids)))

	print(sanity_check)
	print("finished")



	# with open(work_dir + 'ids_dfs_1.txt', 'rb') as f:
	# 	ids_dfs = pickle.load(f)

	# print(len(ids_dfs))

def saving_allfeats():
	filtered_ids = dp.get_filtered_ids()
	ids_allfeats = {}

	for u in range(len(filtered_ids[:5])):
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

def parse_files(ids_dfs, parsed_file):
	full_parsed = {}

	for u in ids_dfs.keys():
		ulog_path = os.path.join(ulog_folder, u + ".ulg")

		dfs = ids_dfs[u]
		names = ids_allfeats[u]

		if len(set(names).intersection(feats_subset)) == len(feats_subset):
			feature_dict = extract_from_tables(dfs, names, feats_subset=feats_subset)
			full_parsed[u] = feature_dict

	return full_parsed

def store_X_and_Y(subset_dir, parsed_file, X_file, y_file):
	with open(subset_dir + "/" + parsed_file, 'rb') as f:
		parsing = pickle.load(f)
	
	full_parsed_split = dp.split_features(parsing)

	y = dp.get_labels(list(parsing.keys()), indexable_meta)

	X = timestamp_bin(full_parsed_split)

	# X = []
	# y = []

	with open(X_file, 'wb') as f:
		pickle.dump(X, f)


	with open(y_file, 'wb') as f:
		pickle.dump(y, f)


def brute_force_feat_select(num_features, best_subset=[]):
	filtered_ids = dp.get_filtered_ids()
	groupings = {}
	id_feats_dict = {}
	all_feature_names = set()

	with open(work_dir + "ids_allfeats.txt", 'rb') as f:
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
						logs_with_27.add(u)
					else:
						groupings[feat] = set([u])
						logs_with_27.add(u)
			except:
				pass

	percentage = 0.6

	all_feature_names_ls = [u for u in all_feature_names_ls if len(groupings[u]) >= percentage * len(filtered_ids)]
	# print(len(all_feature_names_ls))

	feat_mapping = {}

	for i in range(len(all_feature_names_ls)):
		feat_mapping[i] = all_feature_names_ls[i]


	with open(work_dir + "ids_dfs_1.txt", 'rb') as f:
		ids_dfs = pickle.load(f)

	with open(work_dir + "ids_dfs_2.txt", 'rb') as f:
		ids_dfs_2 = pickle.load(f)

	with open(work_dir + "ids_dfs_3.txt", 'rb') as f:
		ids_dfs_3 = pickle.load(f)

	ids_dfs.update(ids_dfs_2)
	ids_dfs.update(ids_dfs_3)

	# ids_dfs = {}

	subset_dir = work_dir + num_features + "_feature_parsings"
	for i in range(len(all_feature_names_ls)):
		if best_subset == []:
			feats_subset = [all_feature_names_ls[i]]
			full_parsed = parse_files(ids_dfs, feats_subset)
			parsed_file = "full_parsed_feat_" + str(i) + ".txt"
		else:
			concat_string = ""
			feats_subset = []
			for j in range(len(best_subset)):
				feats_subset.append(all_feature_names_ls[best_subset[j]])
				concat_string += str(best_subset[j])
				concat_string += "_"

			feats_subset.append(all_feature_names_ls[i])


			full_parsed = parse_files(ids_dfs, feats_subset)

			parsed_file = "full_parsed_feat_" + concat_string + str(i) + ".txt"

		if not os.path.isdir(subset_dir):
			os.makedirs(subset_dir)

		with open(subset_dir + "/" + parsed_file, 'wb') as f:
			pickle.dump(full_parsed, f)


	for i in range(len(all_feature_names_ls)):
		concat_string = ""
		for j in range(len(best_subset)):
			concat_string += str(best_subset[j])
			concat_string += "_"
		parsed_file = "full_parsed_feat_" + concat_string + str(i) + ".txt"
		X_file = subset_dir + "/" + "X_data_feat_" + concat_string + str(i) + ".txt"
		y_file = subset_dir + "/" + "y_data_feat_" + concat_string + str(i) + ".txt"
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

def check_pickle():
	with open(work_dir + "ids_dfs_2.txt", 'rb') as f:
		asdf = pickle.load(f)
	print(len(asdf))


def main():
	# grouping_test()

	# grouping_test_2()

	# timestamp_shorten_test()

	# timestamp_shorten_bme_test()

	# timestamp_interval_test()

	# table_drop_test()

	# checking_multiple_flights()

	# feature_selection_justification()


	pickling_dataframes_test()
	saving_allfeats()	
	# brute_force_feat_select("one")
	# brute_force_feat_select("two", [0, 1])


	# check_pickle()


if __name__ == "__main__":
	main()
