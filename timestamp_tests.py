import data_processing as dp
import pickle

def millis_to_min(millis):
	return millis/60000

def min_to_millis(min):
	return min * 60000

# Temporal Sampling

def get_x_min_chunks(full_parsed, num_mins=20):
	y = []
	full_parsed_copy = copy.deepcopy(full_parsed)
	mins_maxes = get_mins_maxes(full_parsed_copy)
	chunk_size = min_to_millis(num_mins)
	chunked_parse = {}
	ids_intervals = {}

	for log_id in full_parsed_copy.keys():
		t_range = mins_maxes[log_id][1] - mins_maxes[log_id][0]
		num_t_ints = round(t_range/chunk_size)
		chunks = np.linspace(mins_maxes[log_id][0], mins_maxes[log_id][1], num_t_ints)
		ids_intervals[log_id] = chunks
		for i in range(chunks.shape[0] - 1):
			new_id = log_id + ", " + str(i)
			chunked_parse[new_id] = {}
			for topic in full_parsed_copy[log_id].keys():
				timestamps = full_parsed_copy[log_id][topic][0]["timestamp"]
				timestamp_chunk = full_parsed_copy[log_id][topic][0][timestamps.between(chunks[i], chunks[i+1])].reset_index()
				chunked_parse[new_id][topic] = [timestamp_chunk]
				# print(chunked_parse[new_id][topic])   
			if indexable_meta[log_id]["type"] == "Quadrotor":
				y.append(0)
			else:
				y.append(1)
			
	# print(len(chunked_parse))    
	
	return chunked_parse, ids_intervals, y

def timestamp_bin_temporal(chunked_parse, ids_intervals, num_t_ints=50, num_mins=20):
	print("Timestamp Binning")
	X = []

	chunk_size = min_to_millis(num_mins)

	for new_id in chunked_parse.keys():
		log_id = new_id.split(",")[0]
		chunk_index = int(new_id.split(",")[1])
		
		left = ids_intervals[log_id][chunk_index]
		right = ids_intervals[log_id][chunk_index + 1] 
		
		chunk_chunks = np.linspace(left, right, num_t_ints)
		
		# print(chunk_chunks)
		
		X_inst = [[0 for i in range(num_t_ints)] for i in range(len(chunked_parse[new_id]))]

		for i in range(len(chunk_chunks)):
			
			curr_time = int(chunk_chunks[i])
			# print(curr_time)
			index = 0
			for topic in chunked_parse[new_id].keys():
				topic_series = chunked_parse[new_id][topic][0]
				timestamps = topic_series["timestamp"]
				
				if len(timestamps) != 0:

					sample = topic_series[timestamps == curr_time]
					
					topic_name = sample.columns[2]
					if len(sample) == 0:
						# print("reached 1")
						# print(timestamps)
						interp_id = abs(timestamps - curr_time).idxmin()
						# print(interp_id)
						X_inst[index][i] = topic_series.iloc[interp_id][topic_name]
					else:
						# print("reached 2")
						X_inst[index][i] = sample[topic_name].values[0]
					# print(X_inst[index][i])
					
				else:
					X_inst[index][i] = 0
			
				index += 1
#         # print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)) + ", Keep Percentage: " + str(keep_percentage))

#         count += 1        
		X.append(X_inst) 

	# print(np.array(X).shape)

	return X


def temporal_sampling():
	with open("full_parsed_7.txt", 'rb') as f:
		full_parsed = pickle.load(f)
	
	# full_parsed = dict(islice(full_parsed.items(), 1))
	full_parsed_split = dp.split_features(full_parsed)


	chunked_parse, ids_intervals, y = get_x_min_chunks(full_parsed_split) 

	X = timestamp_bin(chunked_parse, ids_intervals)

	return X, y


# Windowed Start, Varied Middle

def timestamp_bin_windowed(full_parsed, num_t_ints=50):
	print("Timestamp Binning")
	X = []

	mins_maxes = dp.get_mins_maxes(full_parsed)

	for new_id, value in full_parsed.items():
		log_id = new_id.split(",")[0]
		chunk_index = int(new_id.split(",")[1])
		
		left = ids_intervals[log_id][chunk_index]
		right = ids_intervals[log_id][chunk_index + 1] 
		
		chunk_chunks = np.linspace(left, right, num_t_ints)
		
		# print(chunk_chunks)
		
		X_inst = [[0 for i in range(num_t_ints)] for i in range(len(chunked_parse[new_id]))]

		for i in range(len(chunk_chunks)):
			
			curr_time = int(chunk_chunks[i])
			# print(curr_time)
			index = 0
			for topic, value in chunked_parse[new_id].items():
				topic_series = chunked_parse[new_id][topic][0]
				timestamps = topic_series["timestamp"]
				
				if len(timestamps) != 0:

					sample = topic_series[timestamps == curr_time]
					
					topic_name = sample.columns[2]
					if len(sample) == 0:
						# print("reached 1")
						# print(timestamps)
						interp_id = abs(timestamps - curr_time).idxmin()
						# print(interp_id)
						X_inst[index][i] = topic_series.iloc[interp_id][topic_name]
					else:
						# print("reached 2")
						X_inst[index][i] = sample[topic_name].values[0]
					# print(X_inst[index][i])
					
				else:
					X_inst[index][i] = 0
			
				index += 1
#         # print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)) + ", Keep Percentage: " + str(keep_percentage))

#         count += 1        
		X.append(X_inst) 

	# print(np.array(X).shape)

	return X

def get_beg_end_chunk(full_parsed, num_mins=20):
	y = []
	full_parsed_copy = copy.deepcopy(full_parsed)
	mins_maxes = get_mins_maxes(full_parsed_copy)
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
			beg_chunk = full_parsed_copy[log_id][topic][0][timestamps.between(beg_left, beg_right)].reset_index()
			end_chunk = full_parsed_copy[log_id][topic][0][timestamps.between(end_left, end_right)].reset_index()
						
			beg_parse[log_id][topic] = [beg_chunk]  
			end_parse[log_id][topic] = [end_chunk]
			
			
	return beg_parse, end_parse

def timestamp_bin_windowed(full_parsed, beg_parse, end_parse, num_mins=20, num_t_ints=50, window_t_ints=10):
	full_parsed_copy = copy.deepcopy(full_parsed)
	mins_maxes = get_mins_maxes(full_parsed_copy)
	chunk_size = min_to_millis(num_mins)
	beg_X = []
	end_X = []
	
	for log_id in full_parsed_copy.keys():
		beg_X_inst = [[0 for i in range(window_t_ints)] for i in range(len(beg_parse[log_id]))]
		end_X_inst = [[0 for i in range(window_t_ints)] for i in range(len(end_parse[log_id]))]
		
		beg_left = mins_maxes[log_id][0]
		beg_right = mins_maxes[log_id][1] + chunk_size
		end_left = mins_maxes[log_id][1] - chunk_size
		end_right = mins_maxes[log_id][1]
		
		beg_chunk_chunks = np.linspace(beg_left, beg_right, window_t_ints)
		end_chunk_chunks = np.linspace(end_left, end_right, window_t_ints)
		
		for i in range(len(beg_chunk_chunks)):
			beg_time = int(beg_chunk_chunks[i])
			end_time = int(end_chunk_chunks[i])
			# print(curr_time)
			index = 0
			for topic in beg_parse[log_id].keys():
				beg_topic_series = beg_parse[log_id][topic][0]
				beg_timestamps = beg_topic_series["timestamp"]
				beg_sample = beg_topic_series[beg_timestamps == beg_time]
				
				end_topic_series = end_parse[log_id][topic][0]
				end_timestamps = end_topic_series["timestamp"]
				end_sample = end_topic_series[end_timestamps == end_time]
				
				topic_name = beg_sample.columns[2]
				
				# print(beg_time)

				if len(beg_timestamps) != 0:
					if len(beg_sample) == 0:
						# print("reached 1")
						# print(timestamps)
						interp_id = abs(beg_timestamps - beg_time).idxmin()
						# print(interp_id)
						beg_X_inst[index][i] = beg_topic_series.iloc[interp_id][topic_name]
					else:
						# print("reached 2")
						beg_X_inst[index][i] = beg_sample[topic_name].values[0]
						# print(X_inst[index][i])  
				else:
					beg_X_inst[index][i] = 0 
					
				# print(beg_X_inst[index][i])

				if len(end_timestamps) != 0:
					if len(end_sample) == 0:
						# print("reached 1")
						# print(timestamps)
						interp_id = abs(end_timestamps - end_time).idxmin()
						# print(interp_id)
						end_X_inst[index][i] = end_topic_series.iloc[interp_id][topic_name]
					else:
						# print("reached 2")
						end_X_inst[index][i] = end_sample[topic_name].values[0]
						# print(X_inst[index][i])  
				else:
					end_X_inst[index][i] = 0 
					
				index += 1
			print("\n")
				
		beg_X.append(beg_X_inst)
		end_X.append(end_X_inst)
		
	return beg_X, end_X

def windowed_sampling():
	with open("full_parsed_7.txt", 'rb') as f:
		full_parsed = pickle.load(f)

	# full_parsed = dict(islice(full_parsed.items(), 1))
	full_parsed_split = dp.split_features(full_parsed)

	with open("X_data_7.txt", 'rb') as f:
		X = pickle.load(f)
		
	with open("Y_data_7.txt", 'rb') as f:
		y = pickle.load(f)

	beg_parse, end_parse = get_beg_end_chunk(full_parsed_split)
	beg_X, end_X = timestamp_bin_windowed(full_parsed_split, beg_parse, end_parse)

	X_array = np.array(X)

	X_array[:, :, :10] = np.array(beg_X)
	X_array[:, :, 40:] = np.array(end_X)

	return X, y

# Local Window Averaging
def timestamp_bin_local(full_parsed, num_t_ints=50):
	print("Timestamp Binning")
	X = []

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

					inter_values = full_parsed[key][key_2][0].iloc[left:right, 1:].transpose().values.tolist()[0]
					# print(inter_values)
					diff_sum = 0
					
					for i in range(len(inter_values) - 1):
						diff = inter_values[i + 1] - inter_values[i]
						diff_sum += diff                                              

					if diff_sum == 0:
						X_inst[index][i] = 0
					else:
						# print(inter_values)
						diff_avg = diff_sum/(len(inter_values) - 1)
						X_inst[index][i] = diff_avg
						# print(diff_avg)

			index += 1

			# print(temp_dict)
			temp_dict = {}
		print("Timestamp Binned: " + str(count) + "/" + str(len(full_parsed)))

		count += 1        
		X.append(X_inst) 

	# print(np.array(X).shape)

	return X

def local_window_averaging_sampling():
	with open("full_parsed_7.txt", 'rb') as f:
		full_parsed = pickle.load(f)

	with open("Y_data_7.txt", 'rb') as f:
		y = pickle.load(f)

	# full_parsed = dict(islice(full_parsed.items(), 1))
	full_parsed_split = dp.split_features(full_parsed)

	X = timestamp_bin_local(full_parsed_split)

	with open("X_data_lwa.txt", 'wb') as f:
		pickle.dump(X, f)	

	return X, y

if __name__ == "__main__":
	# X, y = temporal_sampling()
	# X, y = windowed_sampling()
	X, y = local_window_averaging_sampling()

