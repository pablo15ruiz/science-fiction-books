from hyperparameters import LABEL_COLUMNS

import pandas as pd

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


df = pd.read_csv('../data/science_fiction_books.csv')

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LinearSVC()))
])

train, test = train_test_split(df, random_state=0, test_size=0.2, shuffle=True)
X_train = train.Description
X_test = test.Description

model.fit(X_train, train[LABEL_COLUMNS])
score = model.score(X_test, test[LABEL_COLUMNS])

print(f'score = {score*100}%')
