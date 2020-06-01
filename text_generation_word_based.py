# word-based text generation
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import sys

print("GPU: ", tf.test.is_gpu_available(cuda_only=True) )
# sys.exit()

def clean(file):
	tokenized = []
	for i in file.readlines():
		lowered = i.lower()
		stripped = ''
		for char in lowered:
			if char in 'qwertyuiopasdfghjklzxcvbnm':
				stripped += char
			elif char in '.,\n':
				stripped += ' ' + char + ' '
			# if char == '\n':
			# 	stripped += '\n'
			else:
				stripped += ' '
		tokenized += stripped.split()
		# print(tokenized)

	return tokenized

def makeEncoderDecoder(tokens):
	encoder = {}
	decoder = []
	c = 0
	for i in sorted(list(set(tokens))):
		encoder[i] = c
		decoder.append(i)
		c += 1
		
	return encoder, decoder

def makeTrainingSequences(toInt, length = 25):
	sequences = []
	for i in range(len(toInt)):
		seq = toInt[i-length:i]
		if seq != []:
			sequences.append(seq)

	# print(sequences[:20])
	
	asarray = np.array(sequences)
	print(asarray.shape)

	return asarray

def makeModel(vocab_size, seq_length):
	model = Sequential()
	model.add(Embedding(vocab_size, 50, input_length=seq_length))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.5))
	model.add(LSTM(256))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(vocab_size, activation='softmax'))
	print(model.summary())
	print("MODEL INITIALIZED")
	return model

def generate_seq(model, seq_length, seed, n_words):
	result = []
	print("ASDASKDK")
	# print(seed.shape)

	for _ in range(n_words):
		# seed.tolist()
		# print(seed)
		newseed = seed.tolist()[1:]
		# print(newseed)
		# sys.exit()
		encoded = np.array([seed])
		# encoded = pad_sequences
		yhat = model.predict_classes(encoded, verbose = 0)
		out_word = ''
		
		
		result.append(yhat)
		newseed.append(yhat[0])
		# print(newseed)
		seed = np.array(newseed)
	return result

file = open('roswell.txt', encoding = 'utf-8')
cleanedtxt = clean(file)
print("total tokens, ", len(cleanedtxt))
print("total unique tokens, ", len(list(set(cleanedtxt))))

vocab_size = len(list(set(cleanedtxt))) + 1

encoder, decoder = makeEncoderDecoder(cleanedtxt)

toInt = []
for i in cleanedtxt:
	toInt.append(encoder[i])




gpu = True
train = True
continueTrain = False
seq_size = 10


sequences = makeTrainingSequences(toInt, length=seq_size)
print(sequences.shape)
X, y = sequences[:, :-1], sequences[:,-1]
print(X.shape)
print(y.shape)
print(y)
y = to_categorical(y, num_classes = vocab_size)
seq_length = X.shape[1]

model = makeModel(vocab_size, seq_length)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


if continueTrain:
	model = load_model('model.h5')
if train:
	if gpu:
		with tf.device('/gpu:0'):

			history = model.fit(X, y, batch_size=128, epochs=125)
	else:
		with tf.device('/cpu:0'):
			model = load_model('model.h5')
			history = model.fit(X, y, batch_size=128, epochs=20)
	model.save('model.h5')

from matplotlib import pyplot as plt
#####

model = load_model('model.h5')
import random
random.seed(10)
for _ in range(10):
	seed = sequences[random.randint(0,len(sequences))][:-1]
	# print(seed)
	seedtext = ''
	for i in seed:
		if decoder[i] not in ',.\n':
			seedtext += ' '+ decoder[i] 
		else:
			seedtext += decoder[i]
	# print(seed.shape)


	res = generate_seq(model, seq_length, seed, 100)

	end = ''
	for i in res:
		# print(i)
		if decoder[i[0]] not in ',.\n':

			end += ' ' + decoder[i[0]]

		else:
			end += decoder[i[0]]
	print('\n')
	print("SEED TEXT: ")
	print(seedtext)
	print("OUTPUT: ")
	print(end)
	print("==================")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()









