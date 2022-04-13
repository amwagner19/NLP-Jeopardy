from preprocessing import preprocessData
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# Read in the data
x_train, x_test, x_dev, y_train, y_test, y_dev = preprocessData()

# Create a pipeline for NB using CountVectorizer, TfidfTransformer, and MultinomialNB
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())])

# Fit the pipeline on the training data
text_clf.fit(x_train, y_train)

# Predict on test data and get accuracy
predicted = text_clf.predict(x_test)
print(np.mean(predicted == y_test))
