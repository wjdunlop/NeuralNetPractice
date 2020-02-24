import torch
from torch import nn
from torch.nn import functional as F

# import for graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# MODEL DEFINITION 
# linear(64) -> tanh -> linear(32) -> dropout -> linear(1) -> sigmoid

class model(nn.Module):
	def __init__(self, size):
		# initialize model with size (matching input vector size)
		self.inSize = size
		super(model, self).__init__()
		self.linear1 = nn.Linear(size, 64)
		self.linear2 = nn.Linear(64, 32)
		self.linear3 = nn.Linear(32, 1)

		self.dropout = nn.Dropout(0.3)
	def forward(self, inp, doDropout = True):
		#TAKES: self, input, doDropout(should perform dropout?)
		z1 = self.linear1(inp)
		a1 = F.tanh(z1)

		z2 = self.linear2(a1)
		if doDropout:
			z2 = self.dropout(z2)
		a2 = F.tanh(z2)

		z3 = self.linear3(a2)
		a3 = F.sigmoid(z3)
		return a3

def test(data, model, end = False, drop = False):
	# TAKES: data: dataset name
	#		 model: the model
	#		 end: if the test is the final evaluatory one
	#		 drop: if dropout should be used
	#load data
	testX, testYs = loadFile(data+'-test.txt')

	#testing forward pass
	checkY = model.forward(testX, doDropout = drop)

	# data wrangling into lists for validation
	vY_hat_pre = [float(t) for t in checkY]
	vY_hat = []

	for item in vY_hat_pre:
		#threshold of .5
		if item > .5:
			vY_hat.append(1.)
		else:
			vY_hat.append(0.)
	vY = [float(t) for t in testYs]

	# count correct guesses per class and guesses per class
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
	# RETURNS tuple of (TORCH TENSOR featureVectors shape (number_data_points, input_vector_size) and
	#					TORCH TENSOR outputs shape (number_data_points, 1)
	#function to load data from files
	read= open(filename)
	listform = read.readlines()
	listform = [x.strip() for x in listform]

	#each line is an input, after colon is output
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

		featureVectors.append(thisFeatureVector)
		outputs.append(int(thisOutput))



	return torch.tensor(featureVectors, dtype = torch.float32), torch.tensor(outputs, dtype = torch.float32)

#define current dataset
data = 'heart'

#load dataset
X, ys = loadFile(data+'-train.txt')

# count of iterations
ITER = 1000

# initialize model with parameter of length of input
dSize = X.shape[1]
model = model(dSize)

# initialize learning rate, loss fxn, optimizer
learning_rate = 1e-4
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# learning rate 1e-4, binary cross entropy loss and Adam optimization

# initialize diagnostic graph lists
losses = []
accuracies = []

for i in range(ITER):

	# training forward pass
	y_hat = model.forward(X, doDropout = True).squeeze()

	# calculate loss on this set of training predictions
	thisLoss = loss(y_hat, ys)

	# add to be plotted at end
	losses.append(thisLoss)

	# test to determine accuracy after this number of iterations, add to be plotted at end
	accuracy = test(data, model, drop = False)
	accuracies.append(accuracy)

	# printing loss, accuracy every 100 iters
	if i%100 == 0:
		print(thisLoss, '  iter ', i)
		print(i, ' / ',ITER)
		print('Test set accuracy: ',accuracy)

	# zero gradients, backprop, take optimizer step
	optimizer.zero_grad()
	thisLoss.backward()
	optimizer.step()

	

# test with printing on, final test!
test(data, model, end = True, drop = False)

# plotting loss and accuracy
from matplotlib import pyplot as pyplot

pyplot.subplot(2,1,1)
pyplot.title('LOSS, iteration')
pyplot.plot(losses)
pyplot.subplot(2,1,2)
pyplot.title('ACCURACY, iteration')
pyplot.plot(accuracies)
pyplot.show()

import torchviz
torchviz. make_dot(y_hat.mean(), params = dict(model.named_parameters())).render('a', format = 'png')
