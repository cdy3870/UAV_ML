import os
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, adjusted_mutual_info_score, roc_auc_score
import pickle
from collections import Counter
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
import random 
from itertools import combinations
import csv

class UAVDataset(Dataset):
	"""
	UAV Dataset class

	...

	Attributes
	----------
	X : list
		Parsed and feature engineered time series data
	y : list
		UAV labels

	Methods
	-------
	len():
		gets the size of the dataset
	getitem(index):
		gets an indexed instance
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y
						
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		return torch.Tensor(self.X[index]), torch.tensor(self.y[index]).float()    

class LSTM(nn.Module):
	"""
	LSTM class

	...

	Attributes
	----------
	input_size : int
		number of instances
	hidden_size : int
		hidden size of LSTM layer
	num_classes : int
		number of classes to predict
	num_layers : int
		number of LSTM layers

	Methods
	-------
	forward():
		forward propagation of LSTM
	"""
	def __init__(self, input_size, hidden_size, num_classes, num_layers):
		super(LSTM, self).__init__()
		self.LSTM = nn.LSTM(input_size=input_size,
							hidden_size=hidden_size,
							num_layers=num_layers,
							batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)
	
	def forward(self, x):
		# print(x[0])
		_, (hidden, _) = self.LSTM(x)
		out = hidden[-1]
		# print(out)
		# print(out.shape)
		out = self.fc(out)
		# print(out)
		# print(out.shape)
		return out

def train(model, train_loader, test_loader, params):
	'''
	Trains the LSTM 

			Parameters:
					model (object): Pytorch model
					train_loader (object): Iterable train loader
					test_loader (object): Iterable test loader
					params (dict): Model and training params

			Returns:
					pred_from_last_epoch (list) : Predictions from the last epoch
	'''
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
	progress_bar = tqdm(range(params["num_epochs"]))
	
	num_correct = 0
	true_size = 0

	pred_from_last_epoch = []
	
	for epoch in progress_bar:
		y_true = []
		y_pred = []
		for phase in ("train", "eval"):
			if phase == "train":
				model.train()
				data_loader = train_loader
			else:
				model.eval()
				data_loader = test_loader
				
			for (_, data) in enumerate(data_loader):
				optimizer.zero_grad()
				inputs = data[0].to(device)
				targets = data[1]
				# print(targets)
				targets = targets.to(device)

				with torch.set_grad_enabled(phase=="train"):
					predictions = model(inputs)
					# print(predictions)

					# print(predictions.shape)
					loss = criterion(predictions, targets.long())
					# print(loss.item())

					if phase == "train":
						loss.backward()
						torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
						optimizer.step()

				out, max_indices = torch.max(predictions, dim=1)

				if phase == "eval":
					y_true += targets.tolist()
					y_pred += max_indices.tolist()

					# print(predictions)
				
				# num_correct += torch.sum(max_indices == targets.long()).item()

				# true_size += targets.size()[0]
				# print(targets.size())

		if phase == "eval" and (epoch + 1) % params["num_epochs"]/4 == 0 :
			counts = Counter(y_pred)
			print(counts)
			# print(confusion_matrix(y_true, y_pred))
			report = classification_report(y_true, y_pred, target_names=['Quadrotor', 'Fixed Wing'], output_dict=True)
			print(classification_report(y_true, y_pred, target_names=['Quadrotor', 'Fixed Wing']))
			auc_score = roc_auc_score(y_true, y_pred)
			print(auc_score)
			# val_accuracy = str(num_correct/true_size)
			# print("Val accuracy: {}".format(val_accuracy))

		if phase == "eval" and epoch == params["num_epochs"] - 1:
			pred_from_last_epoch = y_pred

	return pred_from_last_epoch, report, auc_score, counts



def get_merit_score_dict(all_preds, k = 3):
	'''
	Returns the merit scores of all combinations of size k for the features.

			Parameters:
					all_preds (list): Predictions from each feature and the true labels
					k (int): Number of features we look for

			Returns:
					merit_score_dict (dict): A mapping of combos to merit scores
	'''
	num_total_features = len(all_preds) - 1
	num_cols = num_total_features + 1
	AMI_matrix = [[ 0 for i in range(num_cols)] for j in range(num_total_features)]


	# AMI
	for i in range(len(AMI_matrix)):
		for j in range(len(AMI_matrix[0])):
			AMI_matrix[i][j] = adjusted_mutual_info_score(all_preds[i], all_preds[j])


	# Merit score
	merit_score_dict = {}
	combos = list(combinations([i for i in range(num_total_features)], k))


	for combo in combos:
		pairwise_combos = list(combinations(combo, 2))
		cf = np.average(np.array(AMI_matrix)[:, num_cols - 1][list(combo)])

		sum = 0
		for pair in pairwise_combos:
			sum += AMI_matrix[pair[0]][pair[1]]

		try:
			merit_score = (k*cf)/(math.sqrt(k + k*(k - 1)*(sum/len(pairwise_combos))))
		except:
			merit_score = 0

		merit_score_dict[frozenset(combo)] = merit_score


	merit_score_dict = dict(sorted(merit_score_dict.items(), key=lambda x : x[1], reverse=True))

	return merit_score_dict


def feature_selection_experiment():
	# 'rel_alt', 'batt_temp', 'throttle', 'loc_pos | x', 'loc_pos | y', 'loc_pos | z',
	# 'rpy_angles | roll_body', 'rpy_angles | pitch_body', 'rpy_angles | yaw_body'

	X_data_file = "X_data_7.txt"
	with open(X_data_file, 'rb') as f:
		X = pickle.load(f)

	Y_data_file = "Y_data_7.txt"
	with open(Y_data_file, 'rb') as f:
		y = pickle.load(f)


	num_features = len(X[0])
	# print(num_features)


	X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)
	all_preds = []
	# print(np.array(X_train)[:, 0, :].)

	for i in range(num_features):
		single_feature_train = np.array(X_train_full)[:, i, :]
		X_train = np.expand_dims(single_feature_train, axis = 1).tolist()

		single_feature_test = np.array(X_test_full)[:, i, :]
		X_test = np.expand_dims(single_feature_test, axis = 1).tolist()

		# Standardize data
		X_train, X_test = dp.standardize_data(X_train, X_test)


		# Form datasets and dataloaders
		train_dataset = UAVDataset(X_train, y_train)
		test_dataset = UAVDataset(X_test, y_test)


		train_loader = DataLoader(train_dataset,
								  batch_size=8,
								  shuffle=False)
		test_loader = DataLoader(test_dataset,
								  batch_size=1,
								  shuffle=False)

		# Form model
		input_size = train_dataset.__getitem__(0)[0].shape[1]
		hidden_size = 128
		num_classes = 2
		num_layers = 1
		model = LSTM(input_size=input_size,
					 hidden_size=hidden_size,
					 num_classes=num_classes,
					num_layers=num_layers)

		params = {"lr": 0.001, "num_epochs":100}

		# Train model
		print("------------------------------------ " + "Feature: " + str(i + 1) + " ------------------------------------")
		pred_from_last_epoch = train(model, train_loader, test_loader, params)
		all_preds.append(pred_from_last_epoch)
	
	all_preds.append(y_test)


	for i in range(1, len(all_preds) - 3):
		merit_score_dict = get_merit_score_dict(all_preds, k=i)
		print(merit_score_dict)

def main():      
	# Get feature selected data (compute from scratch)
	# X, y = dp.get_data()

	parser = argparse.ArgumentParser()

	# required arguments
	parser.add_argument("-n", "--n_tables",
						type=int, help="number of tables matched (5: 7 total features, 6: 8 total features, 7, 8: 11 total features)", required=True)
	parser.add_argument("-d", "--description", type=str, help="description of experiment", required=True)

	# arguments no longer need to test
	parser.add_argument("-e", "--n_epochs", type=int, help="number of epochs (divisible by 4)", default=100)
	parser.add_argument("-l", "--learning_rate", type=float, help="learning rate", default=0.001)
	parser.add_argument("-sh", "--shuffle", action="store_true", help="shuffle data again", default=True)

	# train/test split or k-fold cross validation
	parser.add_argument("-t", "--n_trials", type=int, help="number of trials to run")
	parser.add_argument("-s", "--test_size", type=int, help="test set percentage")
	parser.add_argument("-k", "--n_folds", type=int, help="number of folds in k-fold cross validation")
	parser.add_argument("-st", "--stratified_k_fold", action="store_true", help="balanced classes during k-fold")

	# modify x
	parser.add_argument("-p", "--shorten_percent", type=int, help="percentage to keep of timestamps")
	parser.add_argument('-i','--indices', nargs='+', help='select indices from parsed features')
	parser.add_argument('-a','--augment', type=int, help='augment train set (fixed wing) percentage')
	parser.add_argument('-ros','--random_oversampling', type=int, help='random oversampling ratio')
	parser.add_argument('-in','--independent', action="store_true", help='standardize time series independently')
	parser.add_argument('-ints','--intervals', type=int, help='number of timestamp intervals', default=50)

	parser.add_argument('-csv','--csv', type=str, help='csv file name output')


	args = parser.parse_args()

	X, y = dp.get_stored_data(args.n_tables, num_t_ints=args.intervals)

	# If we want data augmentation on full set
	# X, y = dp.get_augmented_data(X, y)

	if args.indices != None:
		X = dp.feature_index(args.n_tables, list(map(int, args.indices)))

	print("------------------------------------ Parameters ------------------------------------")
	print("Description: " + args.description)
	print("Test percentage: {}".format(args.test_size))
	print("Learning rate: {}".format(args.learning_rate))
	print("Number of epochs: {}".format(args.n_epochs))
	print("Number of tables: {}".format(args.n_tables))
	print("Total set size: " + str(len(X)))
	print("Number of features: " + str(len(X[0])))
	print("Number of timestamp intervals: " + str(len(X[0][0])))


	if args.csv != None:
		# if os.path.exists(args.csv):
		#     os.remove(args.csv)

		file_name = args.csv
		with open(file_name, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"])            
			csv_writer.writerow(["Parameters"])			
			csv_writer.writerow(["Description: " + args.description])
			csv_writer.writerow(["Test percentage: {}".format(args.test_size)])
			csv_writer.writerow(["Learning rate: {}".format(args.learning_rate)])
			csv_writer.writerow(["Number of epochs: {}".format(args.n_epochs)])
			csv_writer.writerow(["Number of tables: {}".format(args.n_tables)])
			csv_writer.writerow(["Total set size: " + str(len(X))])
			csv_writer.writerow(["Number of features: " + str(len(X[0]))])
			csv_writer.writerow(["Number of timestamp intervals: " + str(len(X[0][0]))])

	if args.n_trials != None:
		for i in range(args.n_trials):

			# Get train and test splits
			test_size = args.test_size/100

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=args.shuffle)

			# Standardize data
			X_train, X_test = dp.standardize_data(X_train, X_test, independent=args.independent)


			# Form datasets and dataloaders
			train_dataset = UAVDataset(X_train, y_train)
			test_dataset = UAVDataset(X_test, y_test)


			train_loader = DataLoader(train_dataset,
									  batch_size=8,
									  shuffle=False)
			test_loader = DataLoader(test_dataset,
									  batch_size=1,
									  shuffle=False)

			# Form model
			input_size = train_dataset.__getitem__(0)[0].shape[1]
			hidden_size = 128
			num_classes = 2
			num_layers = 1
			model = LSTM(input_size=input_size,
						 hidden_size=hidden_size,
						 num_classes=num_classes,
						num_layers=num_layers)

			params = {"lr": args.learning_rate, "num_epochs":args.n_epochs}

			# Train model
			print("------------------------------------ " + "Trial: " + str(i + 1) + " ------------------------------------")
			train(model, train_loader, test_loader, params)

	else:

		if args.stratified_k_fold:
			print("Running Stratified K-fold with K = " + str(args.n_folds))
			kf = StratifiedKFold(n_splits=args.n_folds)
			splits = kf.split(X, y)
		else:
			print("Running Standard K-fold with K = " + str(args.n_folds))
			kf = KFold(n_splits=args.n_folds)
			splits = kf.split(X)

		fold = 1

		report_stack = []
		auc_stack = []

		for train_index, test_index in splits:
			X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
			y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

			if args.shorten_percent != None:
				X = dp.get_stored_data(args.n_tables, percentage=args.shorten_percent)
				X_test = np.array(X)[test_index]

			if args.augment != None:
				X_train, y_train = dp.get_augmented_data(X_train, y_train, augment_percent=args.augment/100)
				X_train = np.array(X_train)
				y_train = np.array(y_train)

			if args.random_oversampling != None:
				X_train, y_train = dp.random_oversample(X_train, y_train, sample_ratio=args.random_oversampling/100)
				X_train = np.array(X_train)
				y_train = np.array(y_train)


			X_train, X_test = dp.standardize_data(X_train, X_test, independent=args.independent)

			# Form datasets and dataloaders
			train_dataset = UAVDataset(X_train, y_train)
			test_dataset = UAVDataset(X_test, y_test)

			train_loader = DataLoader(train_dataset,
									  batch_size=8,
									  shuffle=False)
			test_loader = DataLoader(test_dataset,
									  batch_size=1,
									  shuffle=False)

			# Form model
			input_size = train_dataset.__getitem__(0)[0].shape[1]
			hidden_size = 128
			num_classes = 2
			num_layers = 1
			model = LSTM(input_size=input_size,
						 hidden_size=hidden_size,
						 num_classes=num_classes,
						num_layers=num_layers)

			params = {"lr": args.learning_rate, "num_epochs":args.n_epochs}

			# # Train model
			print("------------------------------------ " + "Fold: " + str(fold) + " ------------------------------------")
			_, report, auc_score, counts = train(model, train_loader, test_loader, params)


			if args.csv != None:
				file_name = args.csv
				with open(file_name, 'a', newline='') as csvfile:
					csv_writer = csv.writer(csvfile)
					csv_writer.writerow(["\n"])            
					csv_writer.writerow(["Fold : " + str(fold)])
					csv_writer.writerow(["", "Quadrotor", "Fixed Wing"])
					csv_writer.writerow(["Precision", round(report["Quadrotor"]["precision"], 4), round(report["Fixed Wing"]["precision"], 4)])
					csv_writer.writerow(["Recall", round(report["Quadrotor"]["recall"], 4), round(report["Fixed Wing"]["recall"], 4)])
					csv_writer.writerow(["F-score", round(report["Quadrotor"]["f1-score"], 4), round(report["Fixed Wing"]["f1-score"], 4)])
					csv_writer.writerow(["AUROC Score", round(auc_score, 4)])
					csv_writer.writerow(["True Counts", report["Quadrotor"]["support"], report["Fixed Wing"]["support"]])
					csv_writer.writerow(["Pred Counts", counts[0], counts[1]])



				report_stack.append([[report["Quadrotor"]["precision"], report["Fixed Wing"]["precision"]],
								 [report["Quadrotor"]["recall"], report["Fixed Wing"]["recall"]], 
								 [report["Quadrotor"]["f1-score"], report["Fixed Wing"]["f1-score"]]])

				auc_stack.append(auc_score)

			fold += 1


		if args.csv != None:
			means = np.mean(np.array(report_stack), axis=0)
			stds = np.std(np.array(report_stack), axis=0)
			auc_mean = np.mean(np.array(auc_stack))
			auc_std = np.std(np.array(auc_stack))

			with open(file_name, 'a', newline='') as csvfile:
				csv_writer = csv.writer(csvfile)
				csv_writer.writerow(["\n"]) 
				csv_writer.writerow(["Averages"])           
				csv_writer.writerow(["", "Quadrotor", "Fixed Wing"])
				csv_writer.writerow(["Precision", round(means[0][0], 4), round(means[0][1], 4)])
				csv_writer.writerow(["Recall", round(means[1][0], 4), round(means[1][1], 4)])
				csv_writer.writerow(["F-score", round(means[2][0], 4), round(means[2][1], 4)])
				csv_writer.writerow(["AUROC Score", round(auc_mean, 4)])

				csv_writer.writerow(["\n"]) 
				csv_writer.writerow(["Standard Deviations"])           
				csv_writer.writerow(["", "Quadrotor", "Fixed Wing"])
				csv_writer.writerow(["Precision", round(stds[0][0], 4), round(stds[0][1], 4)])
				csv_writer.writerow(["Recall", round(stds[1][0], 4), round(stds[1][1], 4)])
				csv_writer.writerow(["F-score", round(stds[2][0], 4), round(stds[2][1], 4)])
				csv_writer.writerow(["AUROC Score", round(auc_std, 4)])

	# ADDITIONAL EXPERIMENTS

	# Feature selection experiment
	# feature_selection_experiment()


	# Guess comparison experiment
	# y_pred = [0 for i in range(len(y_test))]

	# values = [0 for i in range(20)]
	# values[19] = 1
	# y_pred = []
	# for i in range(len(y_test)):
	# 	guess = random.randint(0, 19)
	# 	y_pred.append(values[guess])

	# print(y_pred)



	# print(classification_report(y_test, y_pred, target_names=['Quadrotor', 'Fixed Wing']))


if __name__ == "__main__":
	main()

