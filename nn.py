import torch
from torch import nn
from torch.nn import functional as F
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

class model(nn.Module):
	def __init__(self, size):
		self.inSize = size
		super(model, self).__init__()
		self.linear1 = nn.Linear(size, 64)
		self.linear2 = nn.Linear(64, 32)
		self.linear3 = nn.Linear(32, 1)

		self.dropout = nn.Dropout(0.3)
	def forward(self, inp):
		z1 = self.linear1(inp)
		a1 = F.tanh(z1)
		z2 = self.linear2(a1)
		z2 = self.dropout(z2)
		a2 = F.tanh(z2)
		z3 = self.linear3(a2)
		a3 = F.sigmoid(z3)
		return a3

def test(data, model, end = False):
	testX, testYs = loadFile(data+'-test.txt')

	checkY = model.forward(testX)

	vY_hat_pre = [float(t) for t in checkY]
	vY_hat = []
	for item in vY_hat_pre:
		if item > .5:
			vY_hat.append(1.)
		else:
			vY_hat.append(0.)
	vY = [float(t) for t in testYs]

	# print(vY, vY_hat)
	correct = [0,0]
	guess = [0,0]
	for idx in range(len(vY)):
		guess[int(vY_hat[idx])] += 1
		if vY[idx] == vY_hat[idx]:
			correct[int(vY_hat[idx])]+=1

	if end:
		print('ACCURACY for class 0:', correct[0]/vY.count(0))
		print('guessed: ', guess[0], 'actual: ', vY.count(0))
		print('ACCURACY for class 1:', correct[1]/vY.count(1))
		print('guessed: ', guess[1], 'actual: ', vY.count(1))
		print('OVERALL: ', sum(correct)/len(vY))

	return sum(correct)/len(vY)

def loadFile(filename):
	read= open(filename)
	listform = read.readlines()
	listform = [x.strip() for x in listform]

	dataPointsToSplit = listform
	for line in read.readlines():
		dataPointsToSplit.append(line)

	featureNum = int(dataPointsToSplit[0])
	dataCount = int(dataPointsToSplit[1])

	featureVectors = []
	outputs = []

	for datapoint in dataPointsToSplit[2:]:
		dp = datapoint.split(':')
		thisFeatureVector = dp[0]
		thisOutput = dp[1]

		thisFeatureVector = thisFeatureVector.split()

		for i in range(len(thisFeatureVector)):
			thisFeatureVector[i] = int(thisFeatureVector[i])

		# thisFeatureVector.insert(0,0)
		featureVectors.append(thisFeatureVector)
		outputs.append(int(thisOutput))



	return torch.tensor(featureVectors, dtype = torch.float32), torch.tensor(outputs, dtype = torch.float32)

data = 'ancestry'
# data = input("data?\n>")
X, ys = loadFile(data+'-train.txt')

ITER = 1000
dSize = X.shape[1]

model = model(dSize)
# model = nn.Sequential()
# model.add_module('linear1', nn.Linear(dSize, 64))
# model.add_module('tanh1', nn.Tanh())
# model.add_module('linear2', nn.Linear(64, 32))
# model.add_module('tanh2', nn.Tanh())
# model.add_module('linear3', nn.Linear(32, 1))
# model.add_module('sigmoid', nn.Sigmoid())

learning_rate = 1e-4
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
losses = []
accuracies = []

for i in range(ITER):

	y_hat = model.forward(X).squeeze()
	# print(y_hat.shape)
	# print(ys.shape)
	thisLoss = loss(y_hat, ys)
	losses.append(thisLoss)
	if i%100 == 0:
		print(thisLoss, '  iter ', i)
		print(i, ' / ',ITER)

	optimizer.zero_grad()
	thisLoss.backward()
	optimizer.step()
	accuracies.append(test(data, model))

test(data, model, end = True)


from matplotlib import pyplot as pyplot

pyplot.subplot(2,1,1)
pyplot.plot(losses)
pyplot.subplot(2,1,2)
pyplot.plot(accuracies)
pyplot.show()

import torchviz
torchviz. make_dot(y_hat.mean(), params = dict(model.named_parameters())).render('a', view = True)
