import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import sklearn
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

def clean_data(*args):

	data_files = args

	for file in data_files:
		new_data = pd.read_csv('data/%s' % file)

		# We can actually create two data sets out of each
		# tournament result rather than the one provided by the data

		X_1 = np.array(new_data.loc[:,'FSP.1':'ST5.1'].copy())
		y_1 = np.array([new_data['Result'].copy()]).reshape(len(new_data['Result']),1)

		X_2 = np.array(new_data.loc[:,'FSP.2':'ST5.2'].copy())
		# multiply by one to store the data as binary rather than boolean
		y_2 = np.array((new_data['Result'] == 0).copy()).reshape(len(new_data['Result']),1)

		X = np.vstack([X_1, X_2])
		y = np.vstack([y_1, y_2])

		try:
			all_X = np.vstack([all_X, X])
			all_y = np.vstack([all_y, y])
		except NameError:
			all_X, all_y = X, y

	return (all_X, all_y)

def impute_matrix(matrix):
	imputer = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=True)
	new_matrix = imputer.fit_transform(matrix)
	return new_matrix

def normalize_data(matrix):
	scaler = sklearn.preprocessing.StandardScaler().fit(matrix)
	new_matrix = scaler.transform(matrix)
	return new_matrix


if __name__ == "__main__":

	X, y = clean_data('AusOpen-men-2013.csv', 'AusOpen-women-2013.csv', 'FrenchOpen-men-2013.csv',
					'FrenchOpen-women-2013.csv', 'USOpen-men-2013.csv', 'Wimbledon-men-2013.csv',
					'Wimbledon-women-2013.csv')

	# 4762 NA values; we remove these through mean imputation
	X = impute_matrix(X)
	
	# plot values 
	plt.hist(X[:,0], bins=50)
	plt.show()

	# mean normalize and feature scale X
	X = normalize_data(X)


	


