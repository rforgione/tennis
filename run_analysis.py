import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from clean_data import clean_data, impute_matrix, normalize_data
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation

X, y = clean_data('AusOpen-men-2013.csv', 'AusOpen-women-2013.csv', 'FrenchOpen-men-2013.csv',
					  'FrenchOpen-women-2013.csv', 'USOpen-men-2013.csv', 'Wimbledon-men-2013.csv',
					  'Wimbledon-women-2013.csv')

# 4762 NA values; we remove these through mean imputation
X = impute_matrix(X)

# mean normalize and feature scale X
X = normalize_data(X)

clf = SGDClassifier(loss='hinge', alpha=.0001)
clf.fit(X, y)
scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print "Accuracy: %0.2f (+/- %02f)" % (scores.mean(), scores.std()*2)



