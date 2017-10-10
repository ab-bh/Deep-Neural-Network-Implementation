import numpy as np
import matplotlib.pyplot as plt
from L_DNN_utils import sigmoid, relu, sigmoid_derivative, relu_derivative

def init_parameters_L(layer_dims):
	# layer_dims = [l0, l1, ...... ,nL]
	np.random.seed(1)
	parameters = {}
	for l in xrange(1,len(layer_dims)):
		parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
		parameters['b'+str(l)] = np.zeros((layer_dims[l],1))

		assert(parameters['W'+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
		assert(parameters['b'+str(l)].shape == (layer_dims[l],1))
	return parameters

def forward_prop_L(A_prev, W, b):
	## return A, ((A_prev, W, b), Z )
	Z = np.dot(W, A_prev) + b
	assert(Z.shape == (W.shape[0], A_prev.shape[1]))
	cache = (A_prev, W, b)
	return Z, cache

def activation_forward_L(A_prev, W, b, activation):
	## returns A, (Z, (A_prev, W, b))

	if activation == "relu":
		Z, linear_cache     = forward_prop_L(A_prev, W, b)
		A, activation_cache = relu(Z) 
	
	if activation == "sigmoid":
		Z, linear_cache     = forward_prop_L(A_prev, W, b)
		A, activation_cache = sigmoid(Z) 

	assert(A.shape == (W.shape[0], A_prev.shape[1]))
	return A, (linear_cache, activation_cache)

def L_model_forward(X, parameters):
	## for L-1 times relu layers
	## last layer sigmoid
	L = len(parameters)//2
	A = X
	caches = [] # to store all necessary (linear_cache, activation_cache ) for backprop
	for l in xrange(1, L):
		A_prev = A
		A, cache = activation_forward_L(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation='relu')
		caches.append(cache)
	AL, cache = activation_forward_L(A, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
	caches.append(cache)

	assert(AL.shape == (1, X.shape[1]))
	return AL, caches

def compute_cost(AL, Y):
	## return cost for last layer
	m = Y.shape[1]
	cost = np.squeeze( -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL))/m ) # [[val]] = val using squeeze
	assert (cost.shape == () )
	return cost

def backprop_L(dZ, cache):
	## return dW, db, dA_prev

	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = np.dot(dZ, A_prev.T)/m
	db = np.sum(dZ, axis=1, keepdims=True)/m
	dA_prev = np.dot(W.T, dZ)

	assert( dW.shape == W.shape)
	assert( db.shape == b.shape)
	assert( dA_prev.shape == A_prev.shape)

	return dA_prev, dW, db

def activation_backward_L(dA, cache, activation):
	# take dA, Z, activation to calculate dZ which inturn will calculate dA_prev, dW, db
	linear_cache, activation_cache = cache
	m = linear_cache[0].shape[1]
	if activation == 'relu':
		dZ = relu_derivative(dA, activation_cache)
		dA_prev, dW, db = backprop_L(dZ, linear_cache)
	if activation == 'sigmoid':
		dZ = sigmoid_derivative(dA, activation_cache)
		dA_prev, dW, db = backprop_L(dZ, linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
	# combine above backprop mechanism
	Y = Y.reshape(AL.shape)
	dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
	L = len(caches)
	grads = {}
	m = AL.shape[1]

	# for the only sigmoid layer
	current_cache = caches[L-1]
	grads['dA'+str(L)], grads['dW'+str(L)], grads['db'+str(L)]  = activation_backward_L(dAL, current_cache, activation='sigmoid')

	#for the relu layers
	for l in xrange(L-2,-1,-1):
		current_cache = caches[l]
		grads['dA'+str(l+1)], grads['dW'+str(l+1)], grads['db'+str(l+1)] = activation_backward_L(grads['dA'+str(l+2)], current_cache, activation='relu')

	return grads

def update_parameters(parameters, grads, learning_rate):
	L = len(parameters)//2
	for l in xrange(1,L+1):
		parameters['W'+str(l)] -= learning_rate * grads['dW'+str(l)] 
		parameters['b'+str(l)] -= learning_rate * grads['db'+str(l)] 

	return parameters

def predict(X, y, parameters):
	m = X.shape[1]
	n = len(parameters)//2

	# forward prop
	probas, caches = L_model_forward(X, parameters)
	p = (probas > 0.5).astype(int)
	print "Accuracy: " + str(np.sum(p==y)/float(m))
	return p