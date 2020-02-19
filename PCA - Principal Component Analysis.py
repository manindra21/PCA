import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def PCA_IRIS():
	"""
	Loading and extracting the data.
	"""
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	# load dataset into Pandas DataFrame
	df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

	features = ['sepal length', 'sepal width', 'petal length', 'petal width']
	# Separating out the features
	x = df.loc[:, features].values
	# Separating out the target
	target = df.loc[:, ['target']].values
	target = target[:, 0]

	"""
	Normalizing the data by substracting the mean.
	"""

	class_length, feature_length = np.shape(x)

	for i in range(feature_length):
		m = np.mean(x[:, i])
		x[:, i] = x[:, i] - m

	print("Input data after subtracting the mean.")
	print(x)
	"""
	Standardizing the data by dividing the standard deviation.
	"""

	for i in range(feature_length):
		x[:, i] = x[:, i] / np.std(x[:, i])

	print("Data after Standardization")
	print(x)

	"""
	Applying PCA on Data
	"""
	# Calculating the Covariance Matrix

	feature_covariance = np.cov(np.transpose(x))

	# Calculating Eigen Vector from Covariance.

	eigenvals, eigenvec = np.linalg.eig(feature_covariance)

	# Finding the index of two maximum Eigen vector.
	vector_magnitude = []
	vector_magnitude_index = []
	for i in range(feature_length):
		vector_magnitude.append(np.linalg.norm(eigenvec[:, i]))

	vector_magnitude_index.append(vector_magnitude.index(max(vector_magnitude)))
	vector_magnitude[vector_magnitude_index[0]] = -1
	vector_magnitude_index.append(vector_magnitude.index(max(vector_magnitude)))

	k_vector = np.stack((eigenvec[:, vector_magnitude_index[0]], eigenvec[:, vector_magnitude_index[1]]))

	# Performing dot product of input data and transpose of the selected eigen vector.

	PCA = np.dot(x, np.transpose(k_vector))
	print("The PCA Computation:")
	print(PCA)

	with plt.style.context('seaborn-whitegrid'):
		plt.figure(figsize=(6, 4))
		for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):
			plt.scatter(PCA[target == lab, 0], PCA[target == lab, 1], label=lab, c=col)
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.show()

	""""
	Whitening Process
	"""

	lambda_pca = np.cov(np.transpose(PCA))
	print("The covariance matrix after applying PCA is")
	print(lambda_pca)

	x_pca_whitening = np.stack((x[:, 0] / np.sqrt(lambda_pca[0, 0]), x[:, 1] / np.sqrt(lambda_pca[1, 1])))
	reduced_dimension_data = np.transpose(x_pca_whitening)
	print("The Final Dimension reduced feature vector:")
	print(np.array(reduced_dimension_data))
	print(np.shape(reduced_dimension_data))

	x_array = np.array(reduced_dimension_data[:, 0])
	y_array = np.array(reduced_dimension_data[:, 1])

	with plt.style.context('seaborn-whitegrid'):
		plt.figure(figsize=(6, 4))
		for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):
			plt.scatter(reduced_dimension_data[target == lab, 0], reduced_dimension_data[target == lab, 1], label=lab, c=col)
		plt.xlabel('Principal Component 1')
		plt.ylabel('Principal Component 2')
		plt.legend(loc='upper left')
		plt.tight_layout()
		plt.show()


def BIRCH():

	url = "http://cs.joensuu.fi/sipu/datasets/birch1.txt"

	df = pd.read_csv('C:\Manindra\\birch1.txt', names=['x', 'y'], sep=' ')
	df = df[df['x'].notna()]
	df = df[df['y'].notna()]
	# # Separating out the features
	x = df.loc[:, 'x'].values
	# # Separating out the target
	y = df.loc[:, 'y'].values

	""" NORMALIZATION: Removing the Mean from data."""
	x_mod = np.array(x) - np.mean(x)
	y_mod = np.array(y) - np.mean(y)
	print("Data after normalization:")
	print(x_mod, y_mod)

	""" STANDARDIZATION: Scaling the Data ~ 1
	
	
	"""
	mod_x = np.array(x_mod) / np.std(x)
	mod_y = np.array(y_mod) / np.std(y)

	plt.xlabel('X')
	plt.ylabel('Y')

	"""PCA: To reduce the Dimensionality of data without losing significant data.
	
	
	"""
	dataset = np.stack((mod_x, mod_y), axis=0)
	print("Adjusted Datset:")
	print(dataset)

	"""Calculating Covariance Matrix from the Standardized data."""
	cov_data = np.cov(dataset)
	print(cov_data)

	"""Calculating Eigen Values and Eigen Vector from Covariance Matrix"""
	eigenvals, eigenvecs = np.linalg.eig(cov_data)
	print("Eigenvalues and Eigenvector of Corresponding Covariance matrix:")
	print(eigenvals, eigenvecs)

	"""Sorting EigenPairs"""
	eig_pairs = [(np.abs(eigenvals[i]), eigenvecs[:, i]) for i in range(len(eigenvals))]

	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	print('Eigenvalues in descending order:')
	for i in eig_pairs:
		print(i[0])

	"""Explained Variance: Selecting how many Eigenvectors required for PCA without losing the data"""
	tot = sum(eigenvals)
	var_exp = [(i / tot) * 100 for i in sorted(eigenvals, reverse=True)]
	cum_var_exp = np.cumsum(var_exp)
	print(cum_var_exp)

	""""Projection Matrix"""
	matrix_w = np.hstack((eig_pairs[0][1].reshape(2, 1)))
	print('Matrix W:\n', matrix_w)

	"""Rotated Matrix: Projection onto the new feature space"""
	rotated_matrix = np.dot(np.transpose(dataset), np.transpose(eigenvecs))
	print("Projected Matrix")
	print(rotated_matrix)

	""" WHITENING: To remove the redundant data from the input.
	
	
	"""
	x = rotated_matrix[:, 0]
	y = rotated_matrix[:, 1]

	rotated_data = np.stack((x, y), axis=0)
	print(rotated_data)

	cov_rotated_matrix = np.cov(rotated_data)
	print("Covariance of rotated matrix")
	print(cov_rotated_matrix)

	whitened_x = np.array(rotated_data[0, :]) / np.sqrt(cov_rotated_matrix[0, 0])
	whitened_y = np.array(rotated_data[1, :]) / np.sqrt(cov_rotated_matrix[1, 1])
	print("Whitened Data is:")
	print(whitened_x, whitened_y)

	plt.scatter(mod_x, mod_y, color='r', label="Normalized Data")
	plt.scatter(whitened_x, whitened_y, color='g', label="Whitened Data")
	plt.legend(loc="upper right")
	plt.show()


PCA_IRIS()

PCA_BIRCH()
