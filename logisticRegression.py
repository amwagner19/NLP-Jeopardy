import numpy as np
from preprocessing import preprocessData
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read in the data
x_train, x_test, y_train, y_test = preprocessData()
x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=329)

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_matrix(x_train, mode='count')
y_train = np.asarray(y_train).astype('float32')
x_dev = tokenizer.texts_to_matrix(x_dev, mode='count')
y_dev = np.asarray(y_dev).astype('float32')
x_test = tokenizer.texts_to_matrix(x_test, mode='count')
y_test = np.asarray(y_test).astype('float32')

model=LogisticRegression(multi_class="auto",random_state=1).fit(x_train,y_train)

dev_pred = model.predict(x_dev)
results=accuracy_score(y_dev,dev_pred)
print(results)
#Accuracy:0.49736286919831224