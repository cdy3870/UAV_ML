import data_processing as dp
import pickle
import bisect
import numpy as np
import copy
from itertools import combinations, islice
import pandas as pd
import json
from collections import Counter
import math
import random
import argparse

ulog_folder = "../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
work_dir = "../../../work/uav-ml/"
official_dir = "../../../work/uav-ml/official_experiments"

json_file = "../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"
with open(json_file, 'r') as inputFile:
	meta_json = json.load(inputFile)
indexable_meta = dp.get_indexable_meta(meta_json)


def micros_to_min(micros):
	'''
	Converts microseconds to minutes

	Parameters:
		millis (float): number of microseconds
	Returns:
		min (float) : number of minutes
	'''

	return micros/6e7


def min_to_micros(min):
	'''
	Converts minutes to microseconds

	Parameters:
		min (float): number of minutes
	Returns:
		micros (float) : number of microseconds
	'''

	return min * 6e7

def secs_to_micros(secs):
	'''
	Converts seconds to microseconds

	Parameters:
		secs (float): number of seconds
	Returns:
		micros (float) : number of microseconds
	'''

	return secs * 1e6


def interpolate_value(feat_name, topic_series, timestamps, curr_time):
	'''
	Returns an interpolated value between two timestamps

	Parameters:
		feat_name (string) : feature of interest
		topic_series (pd.Series) : the full data
		timestamps (pd.Series) : all timestamps
		curr_time (float) : timestamp that we are interpolating from (we need to find closest)
	Returns:
		interpolated (float) : interpolated value
	'''

	interp_id = abs(timestamps - curr_time).idxmin()
	closest_timestamp = topic_series.iloc[interp_id]["timestamp"]



	if closest_timestamp < curr_time:
		try:
			next_timestamp = topic_series.iloc[interp_id + 1]["timestamp"]
			if closest_timestamp == next_timestamp:
				return topic_series.iloc[interp_id][feat_name]
			weighting_1 = (curr_time - closest_timestamp)/(next_timestamp - closest_timestamp)
			weighting_2 = 1 - weighting_1
			value_at_next = topic_series.iloc[interp_id + 1][feat_name]
			value_at_closest = topic_series.iloc[interp_id][feat_name]
			return (weighting_2 * value_at_closest + weighting_1 * value_at_next)
		except:
			return topic_series.iloc[interp_id][feat_name]
	else:
		try:
			prev_timestamp = topic_series.iloc[interp_id - 1]["timestamp"]
			if closest_timestamp == prev_timestamp:
				return topic_series.iloc[interp_id][feat_name]
			weighting_1 = (closest_timestamp - curr_time)/(closest_timestamp - prev_timestamp)
			weighting_2 = 1 - weighting_1    
			value_at_prev = topic_series.iloc[interp_id - 1][feat_name]
			value_at_closest = topic_series.iloc[interp_id][feat_name]
			return (weighting_2 * value_at_closest + weighting_1 * value_at_prev)
		except:
			return topic_series.iloc[interp_id][feat_name]    
	
	
def get_timestamp_chunk(full_parsed_copy, log_id, topic, left, right):
	'''
	Returns chunk of data between two timestamps

	Parameters:
		full_parsed_copy (dict) : mapped data
		log_id (int) : ulog id
		topic (string) : current log topic
		left (float) : left timestamp
		right (float) : right timestamp
		
	Returns:
		timestamp_chunk (pd.Dataframe) :  chunk of data between two timestamps
	'''

	timestamps = full_parsed_copy[log_id][topic][0]["timestamp"]
	timestamp_chunk = full_parsed_copy[log_id][topic][0][timestamps.between(left, right)].reset_index()

	if len(timestamp_chunk) > 0:
		beg_index = timestamps.index[timestamps == timestamp_chunk.iloc[0]["timestamp"]].tolist()[0]
		if beg_index != 0:
			beg_index -= 1
			prev_value = full_parsed_copy[log_id][topic][0].iloc[beg_index]
			timestamp_chunk = pd.concat([pd.DataFrame(prev_value).T, timestamp_chunk])
			
		end_index = timestamps.index[timestamps == timestamp_chunk.iloc[-1]["timestamp"]].tolist()[0]
		if end_index != len(timestamps) - 1:
			end_index += 1
			next_value = full_parsed_copy[log_id][topic][0].iloc[end_index]
			timestamp_chunk = pd.concat([timestamp_chunk, pd.DataFrame(next_value).T])

	return timestamp_chunk.reset_index()


def get_beg_end_chunk(full_parsed, num_mins=20):
	'''
	Returns the beginning and end chunk of a flight

	Parameters:
		full_parsed (dict) : mapped data
		num_mins (int) : number of minutes to chunk from the beginning and end
		
	Returns:
		beg_parse (pd.DataFrame) : beginning chunk of a flight
		end_parse (pd.DataFrame) : end chunk of a flight
	'''

	y = []
	full_parsed_copy = copy.deepcopy(full_parsed)
	mins_maxes = dp.get_mins_maxes(full_parsed_copy)
	chunk_size = min_to_millis(num_mins)
	
	beg_parse = {}
	end_parse = {}
	
	for log_id in full_parsed_copy.keys():
		beg_left = mins_maxes[log_id][0]
		beg_right = mins_maxes[log_id][0] + chunk_size
		end_left = mins_maxes[log_id][1] - chunk_size
		end_right = mins_maxes[log_id][1]
		
		beg_parse[log_id] = {}
		end_parse[log_id] = {}
		
		for topic in full_parsed_copy[log_id].keys():

			timestamps = full_parsed_copy[log_id][topic][0]["timestamp"]
			
			beg_chunk = get_timestamp_chunk(full_parsed_copy, log_id, topic, beg_left, beg_right)  
			end_chunk = get_timestamp_chunk(full_parsed_copy, log_id, topic, end_left, end_right)  

			beg_parse[log_id][topic] = [beg_chunk]  
			end_parse[log_id][topic] = [end_chunk]
			
			
	return beg_parse, end_parse


##### 1. Equal-Width Average Sampling ######
def timestamp_bin_average(full_parsed, num_t_ints=50):
	'''
	Average sampling helper function

	Parameters:
		full_parsed (dict) : mapping of flight ids to data
		num_t_ints (int) : number of intervals

	Returns:
		X (list) : timestamp binned data
	'''

	X = []

	intervals = dp.create_intervals(full_parsed, num_t_ints)


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

	# print(np.array(X).shape)

	return X

def equal_width_average_sampling(parse_path, num_t_ints, X_path):
	'''
	Sampling technique that divides flight into equal width intervals and averages
	values

	Parameters:
		parse_path (string) : path to parse mapping
		num_t_ints (int) : number of intervals
		X_path (string) : path to save sampled data
	'''

	with open(parse_path, 'rb') as f:
		full_parsed = pickle.load(f)

	print(full_parsed)

	full_parsed_split = dp.split_features(full_parsed)


	X = timestamp_bin_average(full_parsed_split, num_t_ints=num_t_ints)

	with open(X_path, 'wb') as f:
		pickle.dump(X, f)	


##### 2. Equal-Width Single Sampling ######

def timestamp_bin_single(full_parsed, num_t_ints = 50):
	'''
	Single sampling helper function

	Parameters:
		full_parsed (dict) : mapping of flight ids to data
		num_t_ints (int) : number of intervals

	Returns:
		X (list) : timestamp binned data
	'''

	print("Timestamp Binning")
	X = []
	
	intervals = dp.create_intervals(full_parsed, num_t_ints)

	count = 0
	for log_id in full_parsed.keys():
		X_inst = [[0 for i in range(num_t_ints)] for i in range(len(full_parsed[log_id]))]
		time_intervals = intervals[log_id]

		index = 0
		

		for topic, value_2 in full_parsed[log_id].items():
			# print(full_parsed[log_id][topic])
			# print(time_intervals)            
			for i in range(len(time_intervals)):
				curr_time = time_intervals[i]

				topic_series = full_parsed[log_id][topic][0]
				timestamps = topic_series["timestamp"]
				sample = topic_series[timestamps == curr_time]
				feat_name = sample.columns[1]   

				# if all(v == 0 for v in list(timestamps)):
				# 	print(feat_name)

				if len(sample) == 0:

					interpolated_value = interpolate_value(feat_name, topic_series, timestamps, curr_time)
					# print("Interpolating: " + str(interpolated_value))
					X_inst[index][i] = interpolated_value
				else:
					# print("Found exact: " + str(sample[feat_name].values[0]))
					X_inst[index][i] = sample[feat_name].values[0]

			# print(X_inst[index])

			index += 1
			   
			# print(temp_dict)
			temp_dict = {}
			
		# print(X_inst[0])
		print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)))

		count += 1        
		X.append(X_inst) 	

	return X


def equal_width_single_sampling(parse_path, num_t_ints, X_path):
	'''
	Single sampling gets exact value at timestamp
	or interpolates value

	Parameters:
		parse_path (string) : path to parse mapping
		num_t_ints (int) : number of intervals
		X_path (string) : path to save sampled data
	'''

	with open(parse_path, 'rb') as f:
		full_parsed = pickle.load(f)

	full_parsed_split = dp.split_features(full_parsed)

	X = timestamp_bin_single(full_parsed_split, num_t_ints)

	with open(X_path, 'wb') as f:
		pickle.dump(X, f)	


##### 3. Equal-Width Window Average Sampling #####
def timestamp_bin_local(full_parsed, num_t_ints=50, window_duration=200):
	'''
	Local window averaging helper function

	Parameters:
		full_parsed (dict) : mapping of flight ids to data
		num_t_ints (int) : number of intervals
		window_duration (int) : size of window averaged over
	'''

	print("Timestamp Binning")
	X = []
	
	
	intervals = dp.create_intervals(full_parsed, num_t_ints)	
	
	window_duration = secs_to_micros(window_duration)
	

	count = 0
	for log_id in full_parsed.keys():
		X_inst = [[0 for i in range(num_t_ints)] for i in range(len(full_parsed[log_id]))]
		time_intervals = intervals[log_id]
		index = 0
		
		for topic, value_2 in full_parsed[log_id].items():
			# print(full_parsed[log_id][topic])
			# print(time_intervals)            
			for i in range(len(time_intervals)- 1):
				curr_time = time_intervals[i]

				topic_series = full_parsed[log_id][topic][0]
				timestamps = topic_series["timestamp"]
				# print((curr_time, curr_time + window_duration))
				temp = topic_series[timestamps.between(curr_time, curr_time + window_duration)]
				
				feat = temp.columns[1]
				avg = np.mean(temp[feat])
				# print(avg)


				if math.isnan(avg):
					# print(index)
					X_inst[index][i] = 0
				else:
					X_inst[index][i] = avg                            

		# print("\n")

		index += 1
		temp_dict = {}
			
		print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)))

		count += 1        
		X.append(X_inst) 
		
	return X

def local_window_averaging_sampling(parse_path, num_t_ints, window_duration, X_path):
	'''
	Local window averaging only averages a designated window after each
	interval instead of averaging over the whole interval   

	Parameters:
		parse_path (string) : path to parse mapping
		num_t_ints (int) : number of intervals
		X_path (string) : path to save sampled data

	Returns:
		X (list) : timestamp binned data
	'''

	with open(parse_path, 'rb') as f:
		full_parsed = pickle.load(f)

	with open("Y_data_7.txt", 'rb') as f:
		y = pickle.load(f)

	full_parsed_split = dp.split_features(full_parsed)

	X = timestamp_bin_local(full_parsed_split, num_t_ints=num_t_ints, window_duration=window_duration)	

	with open(X_path, 'wb') as f:
		pickle.dump(X, f)	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--parse_path', type=str, help='specific parse path', required=True)

	parser.add_argument("-s", "--sampling_type", type=int, help="type of sampling", required=True)
	parser.add_argument("-ni", "--num_t_ints", type=int, help="number of intervals (all methods)", required=True)
	parser.add_argument("-wd", "--window_duration", type=float, help="number of minutes in a window (window method)")

	parser.add_argument("-c", "--chunked", type=bool, help="chunking", default=False)
	parser.add_argument("-cm", "--chunk_mins", type=int, help="minutes per chunk", default=5)


	parser.add_argument('-X','--X_path', type=str, help='X output directory')
	parser.add_argument('-Y','--Y_path', type=str, help='y output directory')

	args = parser.parse_args()

	if args.sampling_type == 1:
		equal_width_average_sampling(args.parse_path, args.num_t_ints, args.X_path)
	elif args.sampling_type == 2:
		equal_width_single_sampling(args.parse_path, args.num_t_ints, args.X_path)
	else:
		local_window_averaging_sampling(args.parse_path, args.num_t_ints, args.window_duration, args.X_path)


# Deprecated sampling technique


##### 4. Chunking Sampling #####

# def get_x_min_chunks(full_parsed, num_mins):
#     '''

#     Parameters:
#     	full_parsed () :
#     	num_mins () :
		
#     Returns:
#     	chunked_parse () :
#     	ids_intervals () :
#     	y () :

#     '''

# 	y = []
# 	full_parsed_copy = copy.deepcopy(full_parsed)
# 	mins_maxes = dp.get_mins_maxes(full_parsed_copy)
# 	chunk_size = min_to_micros(num_mins)
# 	chunked_parse = {}
# 	ids_intervals = {}

# 	count = 0 

# 	for log_id in full_parsed_copy.keys():
# 		t_range = mins_maxes[log_id][1] - mins_maxes[log_id][0]
# 		num_t_ints = round(t_range/chunk_size)
# 		chunks = np.linspace(mins_maxes[log_id][0], mins_maxes[log_id][1], num_t_ints)
# 		ids_intervals[log_id] = chunks
# 		for i in range(len(chunks) - 1):
# 			new_id = log_id + ", " + str(i)
# 			chunked_parse[new_id] = {}
# 			for topic in full_parsed_copy[log_id].keys():
# 				timestamp_chunk = get_timestamp_chunk(full_parsed_copy, log_id, topic, chunks[i], chunks[i + 1])                
# 				chunked_parse[new_id][topic] = [timestamp_chunk]
# 				# print(chunked_parse[new_id][topic])   
# 			if indexable_meta[log_id]["type"] == "Quadrotor":
# 				y.append(0)
# 			else:
# 				y.append(1)


# 		print("Chunked: " + str(count) + "/" + str(len(full_parsed_copy)))

# 		count += 1  
# 		# print(len(chunks))
	
# 	return chunked_parse, ids_intervals, y


# def timestamp_bin_temporal_single(chunked_parse, ids_intervals, num_t_ints=50, chunk_mins=20):
#     '''

#     Parameters:
#     	chunked_parse () :
#     	ids_intervals () :
#     	num_t_ints () :
#     	chunk_mins () :
		
#     Returns:
# 		X () :
#     '''

# 	print("Timestamp Binning")
# 	X = []

# 	chunk_size = min_to_micros(chunk_mins)
# 	count = 0

# 	for new_id in chunked_parse.keys():
# 		log_id = new_id.split(",")[0]
# 		chunk_index = int(new_id.split(",")[1])
		
# 		left = ids_intervals[log_id][chunk_index]
# 		right = ids_intervals[log_id][chunk_index + 1] 
		
# 		chunk_chunks = np.linspace(left, right, num_t_ints)
		
# 		# print(chunk_chunks)
		
# 		X_inst = [[0 for i in range(num_t_ints)] for i in range(len(chunked_parse[new_id]))]

# 		for i in range(len(chunk_chunks)):
			
# 			curr_time = int(chunk_chunks[i])
# 			# print(curr_time)
# 			index = 0
# 			for topic in chunked_parse[new_id].keys():
# 				topic_series = chunked_parse[new_id][topic][0]
# 				timestamps = topic_series["timestamp"]
				
# 				if len(timestamps) != 0:

# 					sample = topic_series[timestamps == curr_time]
					
# 					feat_name = sample.columns[3]
# 					if len(sample) == 0:
# 						# print("reached 1")
# 						# print(timestamps)
# 						interp_id = abs(timestamps - curr_time).idxmin()
# 						# print(interp_id)
# 						X_inst[index][i] = topic_series.iloc[interp_id][feat_name]
# 					else:
# 						# print("reached 2")
# 						X_inst[index][i] = sample[feat_name].values[0]
# 					# print(X_inst[index][i])
					
# 				else:
# 					X_inst[index][i] = 0
			
# 				index += 1
# 		print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)))

# 		count += 1        
# 		X.append(X_inst) 

# 	# print(np.array(X).shape)

# 	return X


# def temporal_sampling(parse_path, sample_type, num_mins, num_t_ints, window_duration, X_path, Y_path):
# 	with open(parse_path, 'rb') as f:
# 		full_parsed = pickle.load(f)


# 	chunked_parse, ids_intervals, y = get_x_min_chunks(full_parsed, num_mins=num_mins) 

# 	if sample_type == 1:
# 		X = timestamp_bin_average(chunked_parse, num_t_ints=num_t_ints)

# 		with open(X_path, 'wb') as f:
# 			pickle.dump(X, f)	

# 		with open(Y_path, 'wb') as f:
# 			pickle.dump(y, f)

# 	elif sampling_type == 2:
# 		X = timestamp_bin_single(chunked_parse, num_t_ints=num_t_ints)
# 		with open(X_path, 'wb') as f:
# 			pickle.dump(X, f)	

# 		with open(Y_path, 'wb') as f:
# 			pickle.dump(y, f)

# 	else:
# 		X = timestamp_bin_local(chunked_parse, window_duration=window_duration)

# 		with open(X_path, 'wb') as f:
# 			pickle.dump(X, f)	

# 		with open(X_path, 'wb') as f:
# 			pickle.dump(y, f)


# def generate_subset():
# 	with open("full_parsed_7_multi.txt", 'rb') as f:
# 		full_parsed = pickle.load(f)	


# 	full_parsed_split = dp.split_features(full_parsed)

# 	ids = list(full_parsed.keys())

# 	new_parsed = dp.get_desired_distribution(full_parsed_split, ids)

# 	ids = list(new_parsed.keys())

# 	distribution = dp.get_distribution(ids)
# 	print(distribution)

# 	with open("subset_X_parsed_7_multi.txt", "wb") as f:
# 		pickle.dump(new_parsed, f)

# 	with open("subset_Y_data_7_multi.txt", "wb") as f:
# 		pickle.dump(new_y, f)


# def timestamp_bin_windowed(full_parsed, beg_parse, end_parse, num_mins=20, num_t_ints=50, window_t_ints=10):
# 	'''
# 	Returns the beginning and end chunk of a flight

# 	Parameters:
# 		full_parsed (dict) : mapped data
# 		num_mins (int) : number of minutes to chunk from the beginning and end
		
# 	Returns:
# 		beg_parse (pd.DataFrame) : beginning chunk of a flight
# 		end_parse (pd.DataFrame) : end chunk of a flight
# 	'''


# 	full_parsed_copy = copy.deepcopy(full_parsed)
# 	mins_maxes = dp.get_mins_maxes(full_parsed_copy)
# 	chunk_size = min_to_millis(num_mins)
# 	beg_X = []
# 	end_X = []
# 	count = 0
	
# 	for log_id in full_parsed_copy.keys():
# 		beg_X_inst = [[0 for i in range(window_t_ints)] for i in range(len(beg_parse[log_id]))]
# 		end_X_inst = [[0 for i in range(window_t_ints)] for i in range(len(end_parse[log_id]))]
		
# 		beg_left = mins_maxes[log_id][0]
# 		beg_right = mins_maxes[log_id][1] + chunk_size
# 		end_left = mins_maxes[log_id][1] - chunk_size
# 		end_right = mins_maxes[log_id][1]
		
# 		beg_chunk_chunks = np.linspace(beg_left, beg_right, window_t_ints)
# 		end_chunk_chunks = np.linspace(end_left, end_right, window_t_ints)
		
# 		for i in range(len(beg_chunk_chunks)):
# 			beg_time = int(beg_chunk_chunks[i])
# 			end_time = int(end_chunk_chunks[i])
# 			# print(curr_time)
# 			index = 0
# 			for topic in beg_parse[log_id].keys():
# 				beg_topic_series = beg_parse[log_id][topic][0]
# 				beg_timestamps = beg_topic_series["timestamp"]
# 				beg_sample = beg_topic_series[beg_timestamps == beg_time]
				
# 				end_topic_series = end_parse[log_id][topic][0]
# 				end_timestamps = end_topic_series["timestamp"]
# 				end_sample = end_topic_series[end_timestamps == end_time]
				
# 				feat_name = beg_sample.columns[3]
				
# 				# print(beg_time)

# 				if len(beg_timestamps) != 0:
# 					if len(beg_sample) == 0:
# 						# print("reached 1")
# 						# print(timestamps)
# 						# print(interp_id)
# 						beg_X_inst[index][i] = interpolate_value(feat_name, beg_topic_series, beg_timestamps, beg_time)
# 					else:
# 						# print("reached 2")
# 						beg_X_inst[index][i] = beg_sample[feat_name].values[0]
# 						# print(X_inst[index][i])  
# 				else:
# 					beg_X_inst[index][i] = 0 
					
# 				# print(beg_X_inst[index][i])

# 				if len(end_timestamps) != 0:
# 					if len(end_sample) == 0:
# 						# print("reached 1")
# 						# print(timestamps)
# 						# print(interp_id)
# 						end_X_inst[index][i] = interpolate_value(feat_name, end_topic_series, end_timestamps, end_time)
# 					else:
# 						# print("reached 2")
# 						end_X_inst[index][i] = end_sample[feat_name].values[0]
# 						# print(X_inst[index][i])  
# 				else:
# 					end_X_inst[index][i] = 0 
					
# 				index += 1
				
# 		beg_X.append(beg_X_inst)
# 		end_X.append(end_X_inst)

# 		print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed_copy)))
# 		count += 1
		
# 	return beg_X, end_X

# def windowed_sampling():

# 	with open("full_parsed_7.txt", 'rb') as f:
# 		full_parsed = pickle.load(f)

# 	# full_parsed = dict(islice(full_parsed.items(), 1))
# 	full_parsed_split = dp.split_features(full_parsed)

# 	with open("X_data_7.txt", 'rb') as f:
# 		X = pickle.load(f)
		
# 	with open("Y_data_7.txt", 'rb') as f:
# 		y = pickle.load(f)

# 	beg_parse, end_parse = get_beg_end_chunk(full_parsed_split)
# 	beg_X, end_X = timestamp_bin_windowed(full_parsed_split, beg_parse, end_parse)

# 	X_array = np.array(X)

# 	X_array[:, :, :10] = np.array(beg_X)
# 	X_array[:, :, 40:] = np.array(end_X)

# 	X = X_array.tolist()

# 	with open("X_data_win.txt", 'wb') as f:
# 		pickle.dump(X, f)	

# 	return X, y


