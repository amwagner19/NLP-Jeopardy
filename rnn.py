import numpy as np
from preprocessing import preprocessData
from keras.preprocessing.text import Tokenizer
from keras import models, layers
from sklearn.model_selection import train_test_split

# Read in the data
x_train, x_test, y_train, y_test = preprocessData()
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=329)

numWords = 7500

# Tokenize training and dev data
tokenizer = Tokenizer(num_words=numWords)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
x_train = tokenizer.texts_to_matrix(x_train, mode='binary')
y_train = np.asarray(y_train).astype('float32')
x_dev = tokenizer.texts_to_matrix(x_dev, mode='binary')
y_dev = np.asarray(y_dev).astype('float32')

# Determine the layers of the model
model = models.Sequential()
model.add(layers.Embedding(input_dim=numWords, output_dim=32))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model with accuracy metric and fit on the training/validation data
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1, validation_data=(x_dev, y_dev))

# Tokenize test data and evaluate the model
x_test = tokenizer.texts_to_matrix(x_test, mode='binary')
y_test = np.asarray(y_test).astype('float32')
results = model.evaluate(x_test, y_test)
print(results)

outfile = open("rnnresults.txt",'w')
outfile.write(str(results))
outfile.close()
