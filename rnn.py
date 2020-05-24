#rnn.py
import torch
import torch.nn as nn
import numpy as np

data_dir = 'data/names/'
n_letters = 27
c = 0

encoder = {}
decoder = {}

for i in 'qwertyuiopasdfghjklzxcvbnm':
	encoder[i] = c
	decoder[c] = i
	c += 1

def load_dataset(dataset, isCuda = False):
	dataSet = open(data_dir+dataset+'.txt')
	data = []
	for p in dataSet.readlines():

		data.append(p[:-1].lower())
	return data

def dataToTensors(data):

	dTensors = []
	
	for d in data:
		k = []

		for char in d:

			if char is not ' ':
				da = np.zeros(n_letters)
				da[encoder[char]] = 1
				k.append(da)

		dTensors.append(k)

	return dTensors


data = load_dataset('english', isCuda = False)
print(dataToTensors(data))








