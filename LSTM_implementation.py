import os
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import data_processing as dp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from collections import Counter
import argparse
from sklearn.model_selection import KFold, StratifiedKFold

class UAVDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y
						
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		return torch.Tensor(self.X[index]), torch.tensor(self.y[index]).float()    

class LSTM(nn.Module):
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
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
	progress_bar = tqdm(range(params["num_epochs"]))
	
	num_correct = 0
	true_size = 0
	
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

		if phase == "eval" and (epoch + 1) % (params["num_epochs"]/4) == 0 :
			print(Counter(y_pred))
			# print(confusion_matrix(y_true, y_pred))
			print(classification_report(y_true, y_pred, target_names=['Quadrotor', 'Fixed Wing']))
			# val_accuracy = str(num_correct/true_size)
			# print("Val accuracy: {}".format(val_accuracy))

def main():      
	# Get feature selected data
	# X, y = dp.get_data()

	parser = argparse.ArgumentParser()

	# required arguments
	parser.add_argument("-n", "--n_tables",
	 					type=int, help="number of tables matched (5: 7 total features, 6: 8 total features, 7, 8: 11 total features)", required=True)
	parser.add_argument("-l", "--learning_rate", type=float, help="learning rate", required=True)
	parser.add_argument("-e", "--n_epochs", type=int, help="number of epochs (divisible by 4)", required=True)

	parser.add_argument("-sh", "--shuffle", action="store_true", help="shuffle data again")
	parser.add_argument("-t", "--n_trials", type=int, help="number of trials to run")
	parser.add_argument("-k", "--n_folds", type=int, help="number of folds in k-fold cross validation")
	parser.add_argument("-st", "--stratified_k_fold", action="store_true", help="balanced classes during k-fold")
	parser.add_argument("-s", "--test_size", type=int, help="test set percentage (40)")

	args = parser.parse_args()

	print("------------------------------------ Tested Parameters ------------------------------------")
	print("Number of tables: {}".format(args.n_tables))
	print("Test percentage: {}".format(args.test_size))
	print("Learning rate: {}".format(args.learning_rate))
	print("Number of epochs: {}".format(args.n_epochs))
	print("Shuffle: {}".format(args.shuffle))


	X_data_file = "X_data_" + str(args.n_tables) + ".txt"
	with open(X_data_file, 'rb') as f:
		X = pickle.load(f)

	Y_data_file = "Y_data_" + str(args.n_tables) + ".txt"
	with open(Y_data_file, 'rb') as f:
		y = pickle.load(f)

	# Counter({0: 9255, 1: 359})
	# print(Counter(y))

	# one broken instance
	# X.pop(2898)
	# y.pop(2898)


	print("Total set size: " + str(len(X)))
	print("Number of features: " + str(len(X[0])))
	# print(len(X[0][0]))


	if args.n_trials != None:
		for i in range(args.n_trials):

			# Get train and test splits
			test_size = args.test_size/100

			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=args.shuffle)


			# Standardize data
			X_train = np.array(X_train)
			X_test = np.array(X_test)
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


		for train_index, test_index in splits:
			X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
			y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

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
			print("------------------------------------ " + "Fold: " + str(fold) + " ------------------------------------")
			train(model, train_loader, test_loader, params)
			fold += 1


if __name__ == "__main__":
	main()

