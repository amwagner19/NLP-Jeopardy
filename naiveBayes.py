from preprocessing import preprocessData
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

# Read in the data
x_train, x_test, y_train, y_test = preprocessData()

# Define parameters for grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__norm': ('l1', 'l2'),
    'tfidf__use_idf': (True, False),
    'clf__fit_prior': (True, False),
    'clf__alpha': (1, 1e-1, 1e-2, 1e-3)
}

# Create a pipeline for NB using CountVectorizer, TfidfTransformer, and MultinomialNB
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())])

# Fit the pipeline using grid search on the training data
gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
gs_clf = gs_clf.fit(x_train, y_train)

# Determine best estimator
print(gs_clf.best_estimator_)

# Predict on test data and get accuracy
predicted = gs_clf.best_estimator_.predict(x_test)
print(np.mean(predicted == y_test))
