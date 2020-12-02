import random
from keras import Sequential
from keras import layers
import numpy as np

x = []
y = []

_range = 10
_range2 = 4
size = 10000

for i in range(size):
    a = random.randint(1, _range)
    b = random.randint(1, _range2)
    x.append([a, b])
    y.append([a**b])

x = np.array(x)
y = np.array(y)

print(x)
print(y)
print(x.shape)
print(y.shape)

model = Sequential()
model.add(layers.Dense(16, input_shape = (2,), activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(x = x, y = y, batch_size = 16, epochs = 100)
print(model.predict([[2,2]]))