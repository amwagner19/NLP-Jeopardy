import numpy as np
from preprocessing import preprocessData
from keras.preprocessing.text import Tokenizer
from keras import models, layers
from sklearn.model_selection import train_test_split

# Read in the data
x_train, x_test, y_train, y_test = preprocessData()
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=329)

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
x_train = tokenizer.texts_to_matrix(x_train, mode='binary')
y_train = np.asarray(y_train).astype('float32')
x_dev = tokenizer.texts_to_matrix(x_dev, mode='binary')
y_dev = np.asarray(y_dev).astype('float32')

model = models.Sequential()

model.add(layers.Embedding(input_dim=2000, output_dim=128))

model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(2))
model.add(layers.Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=1, validation_data=(x_dev, y_dev))

x_test = tokenizer.texts_to_matrix(x_test, mode='binary')
y_test = np.asarray(y_test).astype('float32')
results = model.evaluate(x_test, y_test)
print(results)

#[0.6931219696998596, 0.5035396218299866]