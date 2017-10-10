import numpy as np

def sigmoid(Z):
	A = 1/(1+np.exp(-Z))
	return A, Z

def  relu(Z):
	A = np.maximum(0,Z)
	return A, Z

def sigmoid_derivative(dA, activation_cache):
	Z = activation_cache
	g = 1/(1+np.exp(-Z))
	dZ = dA * g * (1-g) 
	assert (dZ.shape == activation_cache.shape)
	return dZ

def relu_derivative(dA, activation_cache):
	Z = activation_cache
	dZ = np.array(dA, copy=True)
	dZ[Z<=0] = 0
	assert (dZ.shape == Z.shape)
	return dZ