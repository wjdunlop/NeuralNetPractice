import numpy as np
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import Sequential
import tensorflow as tf
tf.device('/cpu:0')
def load_file(fn):
    with open(fn) as file:
        return [''.join([ch.lower() for ch in ln]+[' ']) for ln in file.readlines()]

def create_training_examples(lines, x_size):
    X = []
    Y = []
    vocab = []
    for line in lines:
        for idx in range(len(line)-(x_size+1)):
            X.append([ch for ch in line[idx:idx+x_size]])
            Y.append(line[idx+x_size])
        for char in line:
            vocab.append(char)
    
    return X, Y, set(list(vocab))

def make_encoder_decoder(vocab):
    c = 0
    encoder = {}
    decoder = {}
    for ch in vocab:
        encoder[ch] = c
        decoder[c] = ch
        c += 1 
    return encoder, decoder

def data_transform(raw_X, raw_Y, encoder):
    
    X = [[encoder[ch] for ch in ln] for ln in raw_X]
    Y = [encoder[ch] for ch in raw_Y]
    return np.array(X), np.array(Y)

def make_model(vocab_size, x_size):

    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=x_size))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    return model

def generate(model, X, x_size):
    res = []
    for _ in range(10):
        
        seed = np.array([X[np.random.randint(0, X.shape[0])]])
        out = seed[0].tolist()
        # sys.exit()
        for i in range(50):
            new = np.argmax(model.predict(seed), axis = -1)
            out.append(int(new))
            seed = np.array([out[-x_size:]])
            # print(new)
            # print(seed)
        res.append(out)
    return res

def decode(res, decoder, x_size):
    out = []
    for entry in res:
        s = ''.join([decoder[i] for i in entry])
        s = s[:x_size]+"*"+s[x_size:]
        out.append(s)
    return out





lines = load_file('names.txt')
x_size = 2
raw_X, raw_Y, vocab = create_training_examples(lines, x_size)
vocab_size = len(vocab)

encoder, decoder = make_encoder_decoder(vocab)

X, Y = data_transform(raw_X, raw_Y, encoder)
model = make_model(vocab_size, x_size)
model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['sparse_categorical_crossentropy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1)

for i in range(25):
    it_mult = 70
    if i != 0:
        model.load_weights('models/model'+str((i-1)*it_mult)+'.h5')
    model.fit(X, Y, batch_size=64, epochs = it_mult, validation_split=0.1, callbacks=[tensorboard_callback])
    model.save_weights('models/model'+str(i*it_mult)+'.h5')
    # pred_me = np.array([X[1]])
    # print(np.argmax(model.predict(pred_me), axis=-1))
    res = decode(generate(model, X, x_size), decoder, x_size)
    f = open('written/output'+str(it_mult*i)+'.txt', 'w')
    for line in res:
        
        f.write(line)
    f.close()

