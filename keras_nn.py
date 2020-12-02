import numpy as np
from keras import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf
from matplotlib import pyplot as pp

mixed = False

if mixed:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)


def loadFile(filename):
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

    return np.array(featureVectors), np.array(outputs)

def make_model(x):
    model = Sequential()
    print("INPUT SIZE: ", x.shape[1])
    model.add(Input(shape = (x.shape[1], )))
    model.add(Dense(200))
    model.add(Dense(200))
    model.add(Dense(200))
    model.add(Dense(200))
    model.add(Dense(1))
    model.summary()
    return model

do = 'netflix'

x, y = loadFile(do + '-train.txt')
val_x, val_y = loadFile(do + '-test.txt')

model = make_model(x)

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
print("NUM EX: ", x.shape[0])
print("MIXED PRECISION: ", mixed)
history = model.fit(x,y,batch_size=128, epochs = 200).history

# pp.plot(history['loss'])
# pp.show()

print(model.evaluate(val_x, val_y))