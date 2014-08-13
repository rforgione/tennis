import numpy as np
import pandas as pd
import sklearn

data_files = ['AusOpen-men-2013.csv', 'AusOpen-women-2013.csv', 'FrenchOpen-men-2013.csv',
			'FrenchOpen-women-2013.csv', 'USOpen-men-2013.csv', 'Wimbledon-men-2013.csv',
			'Wimbledon-women-2013.csv']

all_data = np.array([[]])

for file in data_files:
	new_data = pd.read_csv('data/%s' % file)

	# We can actually create two data sets out of each
	# tournament result rather than the one provided by the data

	X_1 = np.array(new_data.loc[:,'FSP.1':'ST5.1'].copy())
	y_1 = np.array([new_data['Result'].copy()]).reshape(len(new_data['Result']),1)

	X_2 = np.array(new_data.loc[:,'FSP.2':'ST5.2'].copy())
	# multiply by one to store the data as binary rather than boolean
	y_2 = np.array((new_data['Result'] == 0).copy()).reshape(len(new_data['Result']),1)

	data_1 = np.hstack([X_1, y_1])
	data_2 = np.hstack([X_2, y_2])

	data = np.vstack([data_1, data_2])

	try:
		all_data = np.vstack([all_data, data])
	except ValueError:
		all_data = data

print all_data.shape



